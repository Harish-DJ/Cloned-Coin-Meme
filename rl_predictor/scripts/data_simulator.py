import numpy as np
import pandas as pd

class TrendDataSimulator:
    """
    Simulates meme coin social signals and corresponding price trends.
    Used for training the RL agent without a real-time data pipeline.
    """
    def __init__(self, n_steps=1000, seed=42):
        np.random.seed(seed)
        self.n_steps = n_steps
        self.generate_data()

    def generate_data(self):
        # 1. Base Sentiment (-1 to +1)
        sentiment = np.random.uniform(-0.2, 0.2, self.n_steps)
        
        # 2. Mention Count (with occasional spikes)
        mentions = np.random.poisson(50, self.n_steps).astype(float)
        
        # 3. Price/Trend (The "Actual" label we want to predict)
        # We'll create some "Pump" events
        price = np.ones(self.n_steps)
        for _ in range(self.n_steps // 100):
            start = np.random.randint(50, self.n_steps - 50)
            duration = np.random.randint(5, 15)
            # Increase sentiment and mentions before price pump
            sentiment[start-5:start] += 0.5
            mentions[start-5:start] *= 3
            price[start:start+duration] *= (1.0 + np.random.uniform(0.05, 0.2))

        # Calculate actual trend labels (0=Down, 1=Neutral, 2=Up)
        # Based on price change in next 3 steps
        returns = pd.Series(price).pct_change(3).shift(-3).fillna(0)
        labels = np.ones(self.n_steps, dtype=int)
        labels[returns > 0.02] = 2
        labels[returns < -0.02] = 0

        # Feature Engineering for the 10 required features
        df = pd.DataFrame({
            'sentiment_score': sentiment,
            'mention_count': mentions,
        })
        
        df['mention_growth_rate'] = df['mention_count'].pct_change().fillna(0)
        df['engagement_score'] = df['mention_count'] * np.random.uniform(1, 5, self.n_steps) # Simulated likes/RTs
        df['spike_score'] = (df['mention_count'] > df['mention_count'].rolling(10).mean() * 2).astype(float)
        df['cross_platform_score'] = np.random.randint(1, 6, self.n_steps)
        df['momentum_score'] = df['sentiment_score'].rolling(5).mean().fillna(0)
        df['influencer_score'] = np.random.uniform(0, 1, self.n_steps)
        df['keyword_hype_score'] = np.random.uniform(0, 1, self.n_steps)
        df['volatility_proxy'] = df['mention_count'].rolling(10).std().fillna(0)
        
        # Normalize features to [0, 1] or similar range for RL
        feature_cols = [
            'sentiment_score', 'mention_count', 'mention_growth_rate', 
            'engagement_score', 'spike_score', 'cross_platform_score', 
            'momentum_score', 'influencer_score', 'keyword_hype_score', 
            'volatility_proxy'
        ]
        
        for col in feature_cols:
            if col == 'sentiment_score': continue # keep -1 to 1
            max_val = df[col].max()
            if max_val > 0:
                df[col] = df[col] / max_val

        self.features = df[feature_cols].values
        self.labels = labels
        self.prices = price

    def get_step(self, t):
        if t >= self.n_steps:
            return None, None
        return self.features[t], self.labels[t]

if __name__ == "__main__":
    sim = TrendDataSimulator(100)
    feat, label = sim.get_step(50)
    print(f"Features at t=50: {feat}")
    print(f"Label at t=50 (Next trend): {label}")
