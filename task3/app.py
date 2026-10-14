
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Matrix Operations Tool",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Matrix Operations Tool")
st.write(
    "Enter your matrices and perform mathematical "
    "operations using Python and NumPy."
)

st.info(
    "Enter each matrix row on a new line. "
    "Separate numbers within a row using commas."
)

# Matrix input function
def parse_matrix(matrix_text):
    try:
        rows = [
            line.strip()
            for line in matrix_text.strip().splitlines()
            if line.strip()
        ]

        if not rows:
            raise ValueError("Please enter a matrix.")

        matrix = [
            [float(value.strip()) for value in row.split(",")]
            for row in rows
        ]

        # Ensure all rows have equal columns
        if not all(len(row) == len(matrix[0]) for row in matrix):
            raise ValueError(
                "Every row must contain the same number of values."
            )

        return np.array(matrix, dtype=float)

    except ValueError as error:
        raise ValueError(f"Invalid matrix input: {error}") from error


# Example matrices
default_a = "1, 2\n3, 4"
default_b = "5, 6\n7, 8"

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Matrix A")
    matrix_a_text = st.text_area(
        "Enter Matrix A",
        value=default_a,
        height=130,
        key="matrix_a"
    )

with col2:
    st.subheader("Matrix B")
    matrix_b_text = st.text_area(
        "Enter Matrix B",
        value=default_b,
        height=130,
        key="matrix_b"
    )

st.caption("Example: 1, 2 on the first line and 3, 4 on the next line.")

operation = st.selectbox(
    "Select an operation",
    [
        "Addition (A + B)",
        "Subtraction (A - B)",
        "Multiplication (A × B)",
        "Transpose of A",
        "Transpose of B",
        "Determinant of A",
        "Determinant of B"
    ]
)

if st.button("Calculate Result", type="primary"):
    try:
        A = parse_matrix(matrix_a_text)

        # Display entered matrix
        st.subheader("Input Matrix A")
        st.dataframe(
            pd.DataFrame(A),
            use_container_width=True,
            hide_index=True
        )

        if operation in [
            "Addition (A + B)",
            "Subtraction (A - B)",
            "Multiplication (A × B)",
            "Transpose of B",
            "Determinant of B"
        ]:
            B = parse_matrix(matrix_b_text)

        # Perform selected operation
        if operation == "Addition (A + B)":
            if A.shape != B.shape:
                st.error(
                    "Addition requires both matrices to have "
                    "the same dimensions."
                )
            else:
                result = np.add(A, B)
                st.success("Addition completed!")

        elif operation == "Subtraction (A - B)":
            if A.shape != B.shape:
                st.error(
                    "Subtraction requires both matrices to have "
                    "the same dimensions."
                )
            else:
                result = np.subtract(A, B)
                st.success("Subtraction completed!")

        elif operation == "Multiplication (A × B)":
            if A.shape[1] != B.shape[0]:
                st.error(
                    "Multiplication is not possible. "
                    "Columns of A must equal rows of B."
                )
            else:
                result = np.matmul(A, B)
                st.success("Multiplication completed!")

        elif operation == "Transpose of A":
            result = np.transpose(A)
            st.success("Transpose of A completed!")

        elif operation == "Transpose of B":
            result = np.transpose(B)
            st.success("Transpose of B completed!")

        elif operation == "Determinant of A":
            if A.shape[0] != A.shape[1]:
                st.error(
                    "Determinant can only be calculated "
                    "for a square matrix."
                )
            else:
                result = np.linalg.det(A)
                st.success("Determinant of A calculated!")

        elif operation == "Determinant of B":
            if B.shape[0] != B.shape[1]:
                st.error(
                    "Determinant can only be calculated "
                    "for a square matrix."
                )
            else:
                result = np.linalg.det(B)
                st.success("Determinant of B calculated!")

        # Display result only when an operation succeeded
        if "result" in locals():
            st.subheader("Result")

            if isinstance(result, np.ndarray):
                if result.ndim == 2:
                    result_df = pd.DataFrame(result)
                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.write(result)
            else:
                st.metric(
                    label="Determinant",
                    value=f"{result:.6g}"
                )

    except (ValueError, np.linalg.LinAlgError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Something went wrong: {error}")

st.divider()
st.caption("Built with Python, NumPy, Pandas and Streamlit.")
