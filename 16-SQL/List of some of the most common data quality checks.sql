List of some of the most common data quality checks in my experience:

1. Completeness
— Not null checks on key and non-nullable columns
— Null percentage or ratio checks on nullable columns
— Minimum distinct rows threshold check

2. Consistency
— Schema enforcement
— Cross reference validations
— Integrity violation checks

3. Accuracy
— Precision validations
— Min/Max value checks
— MMM validations (Mean, Median, Mode etc.)
— Type and format validations (Name, Mobile, Address, Date etc.)

4. Uniqueness
— Row level uniqueness checks
— Column level uniqueness checks
— Min/Max uniqueness ratio or percentage checks

5. Validity
— PII validations & checks

6. Lateness
— Max lateness validations
— Lateness percentiles