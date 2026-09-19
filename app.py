
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("student_performance_cleaned.csv")


# =========================================================
# TRAIN MACHINE LEARNING MODEL
# =========================================================

features = [
    "Attendance",
    "Study_Hours",
    "Assignment_Score",
    "Mid1_Score",
    "Mid2_Score",
    "Previous_Semester_Score",
    "Internal_Assessment"
]

X = df[features]
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# =========================================================
# TITLE
# =========================================================

st.title("🎓 Smart Student Performance Analytics")
st.subheader("Performance Prediction and Early-Warning System")

st.write(
    "An academic analytics prototype that analyzes student "
    "performance data, predicts expected final performance, "
    "identifies potential risk factors and provides "
    "academic-support suggestions."
)

st.divider()


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Students", len(df))

with col2:
    st.metric(
        "Average Final Score",
        f"{df['Final_Score'].mean():.2f}"
    )

with col3:
    st.metric(
        "Model R²",
        f"{r2:.2f}"
    )

with col4:
    st.metric(
        "Average Prediction Error",
        f"{mae:.2f}"
    )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    ["🔎 Student Analysis", "📊 Analytics", "🤖 Model"]
)


# =========================================================
# TAB 1 — STUDENT ANALYSIS
# =========================================================

with tab1:

    st.header("🔎 Search and Analyze Student")

    search = st.text_input(
        "Search Student ID",
        placeholder="Example: S0001"
    )

    if search:

        matching_students = df[
            df["Student_ID"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    else:

        matching_students = df.head(20)


    if len(matching_students) == 0:

        st.warning("No student found. Try another Student ID.")

    else:

        selected_student = st.selectbox(
            "Select Student",
            matching_students["Student_ID"].tolist()
        )

        student = df[
            df["Student_ID"] == selected_student
        ].iloc[0]


        st.subheader(f"Student Profile — {selected_student}")


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Attendance",
                f"{student['Attendance']:.0f}%"
            )

        with col2:
            st.metric(
                "Study Hours",
                f"{student['Study_Hours']:.1f}"
            )

        with col3:
            st.metric(
                "Mid-1",
                f"{student['Mid1_Score']:.1f}"
            )

        with col4:
            st.metric(
                "Mid-2",
                f"{student['Mid2_Score']:.1f}"
            )


        # Prepare prediction input

        input_data = pd.DataFrame({
            "Attendance": [student["Attendance"]],
            "Study_Hours": [student["Study_Hours"]],
            "Assignment_Score": [student["Assignment_Score"]],
            "Mid1_Score": [student["Mid1_Score"]],
            "Mid2_Score": [student["Mid2_Score"]],
            "Previous_Semester_Score": [
                student["Previous_Semester_Score"]
            ],
            "Internal_Assessment": [
                student["Internal_Assessment"]
            ]
        })


        predicted_score = model.predict(input_data)[0]

        predicted_score = np.clip(
            predicted_score,
            0,
            100
        )


        # Performance category

        if predicted_score >= 75:

            status = "HIGH PERFORMER"

        elif predicted_score >= 50:

            status = "AVERAGE PERFORMER"

        else:

            status = "NEEDS ATTENTION"


        st.divider()

        st.subheader("🤖 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Final Score",
                f"{predicted_score:.2f}"
            )

        with col2:

            if status == "HIGH PERFORMER":

                st.success(
                    f"✅ {status}"
                )

            elif status == "AVERAGE PERFORMER":

                st.warning(
                    f"⚠️ {status}"
                )

            else:

                st.error(
                    f"🚨 {status}"
                )


        # Risk factors

        st.subheader("🔎 Key Factors")

        factors = []

        if student["Attendance"] < 75:
            factors.append("Low attendance")

        if student["Study_Hours"] < 3:
            factors.append("Low study hours")

        if student["Assignment_Score"] < 60:
            factors.append(
                "Low assignment performance"
            )

        if student["Mid2_Score"] < student["Mid1_Score"]:
            factors.append(
                "Recent examination score has declined"
            )

        if student["Previous_Semester_Score"] < 60:
            factors.append(
                "Low previous semester performance"
            )


        if factors:

            for factor in factors:

                st.write(
                    "🔸",
                    factor
                )

        else:

            st.success(
                "No major risk factors detected."
            )


        # Suggestions

        st.subheader(
            "💡 Suggested Academic Support"
        )

        suggestions = []

        if student["Attendance"] < 75:

            suggestions.append(
                "Monitor attendance regularly"
            )

        if student["Study_Hours"] < 3:

            suggestions.append(
                "Encourage a consistent study schedule"
            )

        if student["Assignment_Score"] < 60:

            suggestions.append(
                "Provide additional assignment support"
            )

        if student["Mid2_Score"] < student["Mid1_Score"]:

            suggestions.append(
                "Review recent examination performance"
            )

        if not suggestions:

            suggestions.append(
                "Continue current academic practices"
            )


        for suggestion in suggestions:

            st.write(
                "✅",
                suggestion
            )


# =========================================================
# TAB 2 — ANALYTICS
# =========================================================

with tab2:

    st.header("📊 Academic Analytics")

    st.subheader(
        "Study Hours vs Final Score"
    )

    st.scatter_chart(
        df,
        x="Study_Hours",
        y="Final_Score"
    )


    st.subheader(
        "Attendance vs Final Score"
    )

    st.scatter_chart(
        df,
        x="Attendance",
        y="Final_Score"
    )


    st.subheader(
        "Assignment Score vs Final Score"
    )

    st.scatter_chart(
        df,
        x="Assignment_Score",
        y="Final_Score"
    )


# =========================================================
# TAB 3 — MODEL
# =========================================================

with tab3:

    st.header("🤖 Machine Learning Model")

    st.write(
        "Algorithm used: Linear Regression"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with col2:

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    with col3:

        st.metric(
            "R²",
            f"{r2:.2f}"
        )


    st.subheader("Features Used")

    st.write(
        ", ".join(features)
    )


    st.subheader(
        "Actual vs Predicted Scores"
    )

    prediction_data = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.scatter_chart(
        prediction_data,
        x="Actual",
        y="Predicted"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Student Performance Analytics and Prediction System | "
    "PBL Prototype"
)

st.caption(
    "Prototype evaluated using a synthetic academic dataset. "
    "Real deployment would require authorized institutional data "
    "and appropriate privacy controls."
)
