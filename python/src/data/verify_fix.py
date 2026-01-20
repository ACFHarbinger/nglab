import pandas as pd

try:
    df = pd.DataFrame({"Price": [1, 2, 3], "Other": [4, 5, 6]})
    print("Original DataFrame:")
    print(df)

    # Simulate the fix in line 77-78
    md = {"id": 0}
    df = df[["Price"]]
    df.columns = [f"Candidate_{md['id']}"]

    print("\nRenamed DataFrame (New Syntax):")
    print(df)

    expected_col = "Candidate_0"
    if df.columns[0] == expected_col:
        print(f"\nSuccess: Column renamed to {expected_col} correctly.")
    else:
        raise ValueError(
            f"Column name mismatch. Expected {expected_col}, got {df.columns[0]}"
        )

except Exception as e:
    print(f"\nError: {e}")
