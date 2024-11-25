import pandas as pd

# Generate 10 example data points without the "country" column
example_data = pd.DataFrame({
    "description": [
        "A vibrant wine with hints of citrus and floral undertones.",
        "Rich and earthy with notes of dark berries and oak.",
        "Bright and refreshing with a crisp finish of green apple.",
        "Full-bodied with layers of black currant and tobacco.",
        "A delicate blend with aromas of peach and honeysuckle.",
        "Spicy and bold with a peppery finish and notes of cherry.",
        "Soft and smooth with hints of vanilla and caramel.",
        "Complex and structured with flavors of plum and cedar.",
        "Light and fresh with a zesty lime and mineral touch.",
        "Sweet and fruity with lingering notes of apricot and honey."
    ],
    "points": [90, 85, 92, 88, 89, 86, 84, 91, 87, 93],
    "price": [25.0, 18.0, 30.0, 40.0, 22.0, 15.0, 20.0, 35.0, 28.0, 45.0],
    "variety": [
        "Sauvignon Blanc", "Cabernet Sauvignon", "Chardonnay", "Merlot",
        "Riesling", "Zinfandel", "Pinot Noir", "Syrah",
        "Verdejo", "Moscato"
    ]
})

# Save the example data to a CSV file for testing
example_file_path = "example_wine_data.csv"
example_data.to_csv(example_file_path, index=False)