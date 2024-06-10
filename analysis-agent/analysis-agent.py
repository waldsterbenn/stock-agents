import subprocess
import pandas as pd
import os
from langchain.tools import tool


def stock_analyzer_tool(ticker: str) -> str:
    """Returns the name of the pickled pandas dataframe, with the results of the techical indicators."""

    # Step 1: Run the Stock Analysis Tool
    script_path = "analysis-agent/ta-analysis-tool.py"
    subprocess.run(["python", script_path, ticker], check=True)

    # Step 2: Assuming the script names the pickle file consistently
    pickle_filename = f"{ticker}_analysis.pkl"

    # Step 3: Load the DataFrame from the pickle file
    if os.path.exists(pickle_filename):
        df = pd.read_pickle(pickle_filename)

        # Perform your analysis on the DataFrame
        # For example, check for specific conditions in the indicators
        # This part is simplified; you'll add your analysis logic here
        # recommendation = "Buy" if some_condition else "Sell"

        # Step 4: Generate and save the report
        report_filename = f"{ticker}_report.txt"
        with open(report_filename, 'w') as report_file:
            report_file.write(f"Analysis for {ticker}:\n")
            report_file.write(f"Recommendation: {recommendation}\n")

        print(f"Report saved to {report_filename}")
        return f"Report saved to {report_filename}"
    else:
        print(f"Failed to find analysis results for {ticker}")
        return f"Failed to find analysis results for {ticker}"


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Example list of tickers
    for ticker in tickers:
        stock_analyzer_tool(ticker)
