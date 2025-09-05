
import matplotlib.pyplot as plt
import pandas as pd

def plot_ticket_trends(df: pd.DataFrame, date_col: str = 'Date of Purchase'):
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"{date_col} not found")
    s = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').value_counts().sort_index()
    plt.figure(figsize=(10,6))
    s.index.astype(str)
    s.plot(kind='line', marker='o')
    plt.title('Customer Support Ticket Trends Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Tickets')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()
