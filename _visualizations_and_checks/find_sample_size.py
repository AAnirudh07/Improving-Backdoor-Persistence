import math

def calculate_sample_size(population_size, z=1.96, margin_of_error=0.05):
    p = 0.5
    e = margin_of_error

    n0 = (z**2 * p * (1 - p)) / (e**2)
    n = n0 / (1 + ((n0 - 1) / population_size))

    return math.ceil(n)

print(calculate_sample_size(1400))              # 95% confidence, 5% margin
print(calculate_sample_size(1400, 1.645, 0.10)) # 90% confidence, 10% margin