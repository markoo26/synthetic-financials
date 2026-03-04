# Fast GMM baseline on S&P 500
uv run learn_ydata_synthetic.py --model gmm --ticker ^GSPC

# CTGAN with more epochs and save output
uv run learn_ydata_synthetic.py --model ctgan --ticker ^GSPC --epochs 500 --output synth_sp500.csv

# Different ticker, custom date range
uv run learn_ydata_synthetic.py --model ctgan --ticker AAPL --start 2018-01-01 --end 2023-12-31

# See all options
uv run learn_ydata_synthetic.py --help
