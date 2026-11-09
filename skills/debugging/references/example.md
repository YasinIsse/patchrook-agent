# Exact edit example

If a test expects `add_numbers(5, 3) == 8`, inspect the implementation first.
An observed `return a - b` can be changed with a unique exact replacement to
`return a + b`. Run the actual test file, rather than the implementation file.
The passing pytest result is evidence for that test, not a guarantee about every input.
