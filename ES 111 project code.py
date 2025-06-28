import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


df_user = pd.read_csv('/content/Store.csv')


df_cleaned = df_user.dropna(subset=["revenue"])
df_cleaned = df_cleaned[df_cleaned["revenue"] > 0]


revenue_data = df_cleaned["revenue"]


mean_revenue = revenue_data.mean()
variance_revenue = revenue_data.var()


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.hist(revenue_data, bins=20, color='skyblue', edgecolor='black')
plt.title('Revenue Distribution (Histogram)')
plt.xlabel('Revenue')
plt.ylabel('Frequency')


plt.subplot(1, 2, 2)

ranges = ['< 20K', '20K-40K', '40K-60K', '60K-80K', '80K+']
revenue_bins = [0, 20000, 40000, 60000, 80000, np.inf]
revenue_categories = pd.cut(revenue_data, bins=revenue_bins, labels=ranges)
revenue_counts = revenue_categories.value_counts()
plt.pie(revenue_counts, labels=ranges, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightpink'])
plt.title('Revenue Distribution (Pie Chart)')


plt.tight_layout()
plt.show()


freq_distribution, bin_edges = np.histogram(revenue_data, bins=revenue_bins)


mean_from_freq = np.average(bin_edges[:-1], weights=freq_distribution)
variance_from_freq = np.average((bin_edges[:-1] - mean_from_freq) ** 2, weights=freq_distribution)


sample_data = revenue_data.sample(frac=0.8, random_state=42)

mean_ci = stats.t.interval(0.95, len(sample_data)-1, loc=sample_data.mean(), scale=stats.sem(sample_data))


variance_ci = stats.chi2.interval(0.95, len(sample_data)-1, loc=sample_data.var(), scale=1)


remaining_data = revenue_data.drop(sample_data.index)


tolerance_interval = np.percentile(remaining_data, [2.5, 97.5])


t_stat, p_value = stats.ttest_1samp(sample_data, 30000)


print("Results:")
print(f"Mean: {mean_revenue}")
print(f"Variance: {variance_revenue}")
print(f"Mean from Frequency Distribution: {mean_from_freq}")
print(f"Variance from Frequency Distribution: {variance_from_freq}")
print(f"95% Confidence Interval for Mean: {mean_ci}")
print(f"95% Confidence Interval for Variance: {variance_ci}")
print(f"95% Tolerance Interval: {tolerance_interval}")
print(f"Hypothesis Test - t-statistic: {t_stat}")
print(f"Hypothesis Test - p-value: {p_value}")