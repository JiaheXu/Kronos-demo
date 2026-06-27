import gc
import functools
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from binance.client import Client

from model import KronosTokenizer, Kronos, KronosPredictor

# --- Configuration ---
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH": "../Kronos_model",
    "OUTPUT_PATH": Path(__file__).parent.resolve() / "results",
    "N_PREDICTIONS": 50,
    "SERVER_HOST": "0.0.0.0",
    "SERVER_PORT": 8000,
    "SYMBOLS": [
        {
            "key": "btc",
            "symbol": "BTCUSDT",
            "label": "BTC/USDT",
        },
        {
            "key": "eth",
            "symbol": "ETHUSDT",
            "label": "ETH/USDT",
        },
    ],
    "TIMEFRAMES": [
        {
            "key": "15m",
            "interval": "15m",
            "label": "15-Minute",
            "hist_points": 360,
            "pred_horizon": 24,
            "vol_window": 24,
            "step": pd.Timedelta(minutes=15),
            "freq": "15min",
            "schedule_minutes": 15,
            "artifact_suffix": "_15m",
            "dashboard_enabled": True,
        },
    ],
    "MODELS": [
        {
            "key": "mini",
            "label": "Kronos-mini",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
            "model_id": "NeoQuasar/Kronos-mini",
        },
        {
            "key": "small",
            "label": "Kronos-small",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "model_id": "NeoQuasar/Kronos-small",
        },
        {
            "key": "base",
            "label": "Kronos-base",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "model_id": "NeoQuasar/Kronos-base",
        },
    ],
}


def artifact_name(prefix, symbol_key, model_key, timeframe_config, extension):
    """Builds a stable artifact filename per symbol/model/timeframe."""
    suffix = timeframe_config["artifact_suffix"]
    if prefix == "forecast":
        return f"{symbol_key}_{model_key}_forecast{suffix}.{extension}"
    return f"{prefix}_{symbol_key}_{model_key}{suffix}.{extension}"


def dashboard_metric_id(symbol_key, timeframe_key, model_key, metric_key):
    """Builds a unique DOM id for a symbol/timeframe/model metric."""
    return f"{symbol_key}-{timeframe_key}-{model_key}-{metric_key}"


def load_models():
    """Loads all configured Kronos models and tokenizers."""
    predictors = {}
    for model_config in Config["MODELS"]:
        print(f"Loading {model_config['label']}...")
        tokenizer = KronosTokenizer.from_pretrained(
            model_config["tokenizer_id"],
            cache_dir=Config["MODEL_PATH"],
        )
        model = Kronos.from_pretrained(
            model_config["model_id"],
            cache_dir=Config["MODEL_PATH"],
        )
        tokenizer.eval()
        model.eval()
        predictors[model_config["key"]] = {
            "config": model_config,
            "predictor": KronosPredictor(model, tokenizer, device="cuda", max_context=512),
        }
        print(f"{model_config['label']} loaded successfully.")
    return predictors


def make_prediction(df, predictor, timeframe_config):
    """Generates probabilistic forecasts using the Kronos model."""
    last_timestamp = df['timestamps'].max()
    start_new_range = last_timestamp + timeframe_config["step"]
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=timeframe_config["pred_horizon"],
        freq=timeframe_config["freq"]
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making main prediction (T=1.0)...")
        begin_time = time.time()
        close_preds_main, volume_preds_main = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=timeframe_config["pred_horizon"], T=1.0, top_p=0.95,
            sample_count=Config["N_PREDICTIONS"], verbose=True
        )
        print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")

        # print("Making volatility prediction (T=0.9)...")
        # begin_time = time.time()
        # close_preds_volatility, _ = predictor.predict(
        #     df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        #     pred_len=Config["PRED_HORIZON"], T=0.9, top_p=0.9,
        #     sample_count=Config["N_PREDICTIONS"], verbose=True
        # )
        # print(f"Volatility prediction completed in {time.time() - begin_time:.2f} seconds.")
        close_preds_volatility = close_preds_main

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_binance_data(symbol_config, timeframe_config):
    """Fetches K-line data from the Binance public API."""
    symbol, interval = symbol_config["symbol"], timeframe_config["interval"]
    limit = timeframe_config["hist_points"] + timeframe_config["vol_window"]

    print(f"Fetching {limit} bars of {symbol} {interval} data from Binance...")
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)

    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)

    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    print("Data fetched successfully.")
    return df


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df, timeframe_config):
    """
    Calculates upside and volatility amplification probabilities for the 24h horizon.
    """
    last_close = hist_df['close'].iloc[-1]

    # 1. Upside Probability (for the 24-hour horizon)
    # This is the probability that the price at the end of the horizon is higher than now.
    final_hour_preds = close_preds_df.iloc[-1]
    upside_prob = (final_hour_preds > last_close).mean()

    # 2. Volatility Amplification Probability (over the 24-hour horizon)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-timeframe_config["vol_window"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(
        f"Upside Probability ({timeframe_config['label']}): {upside_prob:.2%}, "
        f"Volatility Amplification Probability: {vol_amp_prob:.2%}"
    )
    return upside_prob, vol_amp_prob


def create_plot(hist_df, close_preds_df, volume_preds_df, model_config, symbol_config, timeframe_config):
    """Generates and saves a comprehensive forecast chart."""
    model_type = model_config["key"]
    print(
        f"Generating comprehensive forecast chart for "
        f"{symbol_config['label']} {model_type} ({timeframe_config['label']})..."
    )
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime(
        [last_hist_time + timeframe_config["step"] * (i + 1) for i in range(len(close_preds_df))]
    )

    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(
        f'{symbol_config["label"]} {model_type} Forecast '
        f'({timeframe_config["label"]}, Next {timeframe_config["pred_horizon"]} Steps)',
        fontsize=16,
        weight='bold'
    )
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=0.03)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=0.03)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    separator_time = hist_time.iloc[-1] + (timeframe_config["step"] / 2)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / artifact_name(
        "prediction_chart",
        symbol_config["key"],
        model_config["key"],
        timeframe_config,
        "png",
    )
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")
    return chart_path


def save_prediction_results(df_for_model, close_preds_df, volume_preds_df, model_config, symbol_config, timeframe_config):
    """Saves forecast outputs locally for later inspection."""
    print(
        f"Saving forecast outputs for "
        f"{symbol_config['label']} {model_config['label']} ({timeframe_config['label']})..."
    )
    output_dir = Config["OUTPUT_PATH"]
    output_dir.mkdir(parents=True, exist_ok=True)

    last_timestamp = df_for_model['timestamps'].max()
    pred_time = pd.date_range(
        start=last_timestamp + timeframe_config["step"],
        periods=timeframe_config["pred_horizon"],
        freq=timeframe_config["freq"]
    )

    result_df = pd.DataFrame({
        'timestamp': pred_time,
        'close_mean': close_preds_df.mean(axis=1).to_numpy(),
        'close_min': close_preds_df.min(axis=1).to_numpy(),
        'close_max': close_preds_df.max(axis=1).to_numpy(),
        'volume_mean': volume_preds_df.mean(axis=1).to_numpy(),
    })

    csv_path = output_dir / artifact_name(
        "forecast", symbol_config["key"], model_config["key"], timeframe_config, "csv"
    )
    json_path = output_dir / artifact_name(
        "forecast", symbol_config["key"], model_config["key"], timeframe_config, "json"
    )

    result_df.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_df.to_dict(orient='records'), f, indent=2, default=str)

    print(f"Saved forecast table to: {csv_path}")
    print(f"Saved forecast JSON to: {json_path}")
    return csv_path, json_path


def update_html(all_results):
    """Updates the dashboard with the latest metrics and artifact paths."""
    print("Updating index.html...")
    html_path = Config["REPO_PATH"] / 'index.html'
    beijing_tz = timezone(timedelta(hours=8))
    now_beijing_str = datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(
        r'(<strong id="update-time">).*?(</strong>)',
        lambda m: f'{m.group(1)}{now_beijing_str}{m.group(2)}',
        content
    )

    for symbol_key, timeframe_results in all_results.items():
        for timeframe_key, model_results in timeframe_results.items():
            for model_key, result in model_results.items():
                upside_id = dashboard_metric_id(symbol_key, timeframe_key, model_key, "upside-prob")
                vol_id = dashboard_metric_id(symbol_key, timeframe_key, model_key, "vol-amp-prob")
                mean_id = dashboard_metric_id(symbol_key, timeframe_key, model_key, "mean-price")
                csv_id = dashboard_metric_id(symbol_key, timeframe_key, model_key, "csv-link")
                json_id = dashboard_metric_id(symbol_key, timeframe_key, model_key, "json-link")
                chart_id = dashboard_metric_id(symbol_key, timeframe_key, model_key, "chart-img")

                content = re.sub(
                    rf'(<p class="metric-value" id="{upside_id}">).*?(</p>)',
                    lambda m, value=f'{result["upside_prob"]:.1%}': f'{m.group(1)}{value}{m.group(2)}',
                    content
                )
                content = re.sub(
                    rf'(<p class="metric-value" id="{vol_id}">).*?(</p>)',
                    lambda m, value=f'{result["vol_amp_prob"]:.1%}': f'{m.group(1)}{value}{m.group(2)}',
                    content
                )
                content = re.sub(
                    rf'(<p class="metric-value" id="{mean_id}">).*?(</p>)',
                    lambda m, value=f'{result["final_mean_close"]:,.2f}': f'{m.group(1)}{value}{m.group(2)}',
                    content
                )
                content = re.sub(
                    rf'(<a id="{csv_id}" href=").*?(")',
                    lambda m, value=result["csv_href"]: f'{m.group(1)}{value}{m.group(2)}',
                    content
                )
                content = re.sub(
                    rf'(<a id="{json_id}" href=").*?(")',
                    lambda m, value=result["json_href"]: f'{m.group(1)}{value}{m.group(2)}',
                    content
                )
                content = re.sub(
                    rf'(<img id="{chart_id}" src=").*?(")',
                    lambda m, value=result["chart_href"]: f'{m.group(1)}{value}{m.group(2)}',
                    content
                )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML file updated successfully.")


def git_commit_and_push(commit_message):
    """Adds, commits, and pushes specified files to the Git repository."""
    print("Performing Git operations...")
    try:
        os.chdir(Config["REPO_PATH"])
        add_targets = ['index.html']
        for symbol_config in Config["SYMBOLS"]:
            for timeframe_config in Config["TIMEFRAMES"]:
                for model_config in Config["MODELS"]:
                    add_targets.extend([
                        artifact_name(
                            "prediction_chart",
                            symbol_config["key"],
                            model_config["key"],
                            timeframe_config,
                            "png",
                        ),
                        f'results/{artifact_name("forecast", symbol_config["key"], model_config["key"], timeframe_config, "csv")}',
                        f'results/{artifact_name("forecast", symbol_config["key"], model_config["key"], timeframe_config, "json")}',
                    ])
        subprocess.run(['git', 'add', *add_targets], check=True, capture_output=True, text=True)
        commit_result = subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(commit_result.stdout)
        push_result = subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
        print(push_result.stdout)
        print("Git push successful.")
    except subprocess.CalledProcessError as e:
        output = e.stdout if e.stdout else e.stderr
        if "nothing to commit" in output or "Your branch is up to date" in output:
            print("No new changes to commit or push.")
        else:
            print(f"A Git error occurred:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")


class NoCacheRequestHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard files and disables browser caching for live updates."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def start_http_server():
    """Starts a static file server for the generated dashboard."""
    handler = functools.partial(NoCacheRequestHandler, directory=str(Config["REPO_PATH"]))
    server = ThreadingHTTPServer((Config["SERVER_HOST"], Config["SERVER_PORT"]), handler)

    thread = threading.Thread(target=server.serve_forever, name="dashboard-http-server", daemon=True)
    thread.start()

    print(
        f"Serving dashboard from {Config['REPO_PATH']} at "
        f"http://127.0.0.1:{Config['SERVER_PORT']}/"
    )
    return server


def run_symbol_timeframe_task(models, symbol_config, timeframe_config):
    """Executes one full update cycle for a single symbol and timeframe."""
    print(f"\nRunning {symbol_config['label']} {timeframe_config['label']} forecast cycle...")
    df_full = fetch_binance_data(symbol_config, timeframe_config)
    df_for_model = df_full.iloc[:-1]
    hist_df_for_plot = df_for_model.tail(timeframe_config["hist_points"])
    hist_df_for_metrics = df_for_model.tail(timeframe_config["vol_window"])
    model_results = {}

    for model_key, model_bundle in models.items():
        model_config = model_bundle["config"]
        predictor = model_bundle["predictor"]

        close_preds, volume_preds, v_close_preds = make_prediction(df_for_model, predictor, timeframe_config)
        upside_prob, vol_amp_prob = calculate_metrics(
            hist_df_for_metrics, close_preds, v_close_preds, timeframe_config
        )
        chart_path = create_plot(
            hist_df_for_plot, close_preds, volume_preds, model_config, symbol_config, timeframe_config
        )
        csv_path, json_path = save_prediction_results(
            df_for_model, close_preds, volume_preds, model_config, symbol_config, timeframe_config
        )

        model_results[model_key] = {
            "upside_prob": upside_prob,
            "vol_amp_prob": vol_amp_prob,
            "final_mean_close": float(close_preds.mean(axis=1).iloc[-1]),
            "chart_href": chart_path.relative_to(Config["REPO_PATH"]).as_posix(),
            "csv_href": csv_path.relative_to(Config["REPO_PATH"]).as_posix(),
            "json_href": json_path.relative_to(Config["REPO_PATH"]).as_posix(),
        }

        del close_preds, volume_preds, v_close_preds
        gc.collect()

    results_to_return = model_results
    del df_full, df_for_model, hist_df_for_plot, hist_df_for_metrics
    gc.collect()
    return results_to_return


def main_task(models, timeframe_keys=None):
    """Executes one full update cycle."""
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    selected_keys = set(timeframe_keys or [tf["key"] for tf in Config["TIMEFRAMES"]])
    selected_timeframes = [tf for tf in Config["TIMEFRAMES"] if tf["key"] in selected_keys]
    all_results = {}

    for symbol_config in Config["SYMBOLS"]:
        all_results[symbol_config["key"]] = {}
        for timeframe_config in selected_timeframes:
            all_results[symbol_config["key"]][timeframe_config["key"]] = run_symbol_timeframe_task(
                models, symbol_config, timeframe_config
            )

    update_html(all_results)

    commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message)

    print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")


def run_scheduler(models):
    """A continuous scheduler that runs update tasks on the shortest configured cadence."""
    min_schedule_minutes = min(tf["schedule_minutes"] for tf in Config["TIMEFRAMES"])

    while True:
        now = datetime.now(timezone.utc)
        candidate_now_slot = now.replace(second=5, microsecond=0)
        if now.minute % min_schedule_minutes == 0 and candidate_now_slot > now:
            next_run_time = candidate_now_slot
        else:
            next_minute_multiple = (
                ((now.minute // min_schedule_minutes) + 1) * min_schedule_minutes
            )
            next_run_time = now.replace(second=5, microsecond=0)
            if next_minute_multiple >= 60:
                next_run_time = next_run_time.replace(minute=0) + timedelta(hours=1)
            else:
                next_run_time = next_run_time.replace(minute=next_minute_multiple)
        sleep_seconds = (next_run_time - now).total_seconds()

        if sleep_seconds > 0:
            print(f"Current time: {now:%Y-%m-%d %H:%M:%S UTC}.")
            print(f"Next run at: {next_run_time:%Y-%m-%d %H:%M:%S UTC}. Waiting for {sleep_seconds:.0f} seconds...")
            time.sleep(sleep_seconds)

        try:
            due_timeframes = [
                tf["key"]
                for tf in Config["TIMEFRAMES"]
                if next_run_time.minute % tf["schedule_minutes"] == 0
            ]
            main_task(models, timeframe_keys=due_timeframes)
        except Exception as e:
            print(f"\n!!!!!! A critical error occurred in the main task !!!!!!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Retrying in 5 minutes...")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            time.sleep(300)


if __name__ == '__main__':
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)
    Config["OUTPUT_PATH"].mkdir(parents=True, exist_ok=True)

    http_server = start_http_server()
    loaded_models = load_models()
    try:
        main_task(loaded_models)  # Run once on startup
        run_scheduler(loaded_models)  # Start the schedule
    finally:
        http_server.shutdown()
        http_server.server_close()
