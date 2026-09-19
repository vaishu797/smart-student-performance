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
    page_title="Smart Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("student_performance_cleaned.csv")


# =========================================================
# MACHINE LEARNING MODEL
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
# PERFORMANCE CATEGORIES
# =========================================================

def performance_category(score):

    if score >= 75:
        return "High Performer"

    elif score >= 50:
        return "Average Performer"

    else:
        return "Needs Attention"


df["Performance_Category"] = df["Final_Score"].apply(
    performance_category
)

high_count = (
    df["Performance_Category"] == "High Performer"
).sum()

average_count = (
    df["Performance_Category"] == "Average Performer"
).sum()

attention_count = (
    df["Performance_Category"] == "Needs Attention"
).sum()


# =========================================================
# FIND STUDENT
# =========================================================

def find_student(student_id):

    student_id = str(student_id).strip().upper()

    matches = df[
        df["Student_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
        == student_id
    ]

    if len(matches) > 0:
        return matches.iloc[0]

    return None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎓 Smart Student")

    st.caption("Performance Analytics System")

    st.divider()

    st.markdown("### 🧭 Project Pipeline")

    st.write("📥 Academic Data")
    st.write("🧹 Preprocessing")
    st.write("📊 Analytics")
    st.write("🤖 Prediction")
    st.write("🚦 Early Warning")
    st.write("💡 Academic Support")

    st.divider()

    st.markdown("### 📌 Prototype")

    st.caption(
        "This prototype uses a synthetic academic dataset "
        "for demonstration."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🎓 Smart Student Performance Analytics")

st.subheader("Predict • Identify • Improve")

st.write(
    "A data-driven academic analytics and early-warning "
    "system for student performance monitoring."
)

st.divider()


# =========================================================
# TOP KPI METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "👨‍🎓 Total Students",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "📊 Average Final Score",
        f"{df['Final_Score'].mean():.2f}"
    )

with col3:

    st.metric(
        "🤖 Model R²",
        f"{r2:.2f}"
    )

with col4:

    st.metric(
        "📉 Average Prediction Error",
        f"{mae:.2f}"
    )


# =========================================================
# PERFORMANCE OVERVIEW
# =========================================================

st.header("📊 Academic Performance Overview")

p1, p2, p3 = st.columns(3)

with p1:

    st.success(
        f"🟢 High Performers\n\n"
        f"**{high_count} students**"
    )

with p2:

    st.warning(
        f"🟡 Average Performers\n\n"
        f"**{average_count} students**"
    )

with p3:

    st.error(
        f"🔴 Needs Attention\n\n"
        f"**{attention_count} students**"
    )


# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Student Analysis",
        "📊 Analytics",
        "🤖 Prediction Model",
        "🏫 About System"
    ]
)


# =========================================================
# TAB 1 — STUDENT ANALYSIS
# =========================================================

with tab1:

    st.header("🔎 Student Analysis")

    st.write(
        "Search any student using the Student ID from the dataset."
    )

    search_col, demo_col = st.columns([4, 1])

    with search_col:

        search_id = st.text_input(
            "Student ID",
            value=st.session_state.get(
                "demo_student",
                ""
            ),
            placeholder="Example: S0500"
        )

    with demo_col:

        st.write("")

        demo_clicked = st.button(
            "🎬 Demo Student",
            use_container_width=True
        )

    if demo_clicked:

        st.session_state["demo_student"] = "S0500"

        st.rerun()


    # -----------------------------------------------------
    # FIND STUDENT
    # -----------------------------------------------------

    student = None

    if search_id:

        student = find_student(search_id)


    # -----------------------------------------------------
    # STUDENT FOUND
    # -----------------------------------------------------

    if student is not None:

        st.success(
            f"Student {student['Student_ID']} found."
        )

        st.subheader("👤 Student Profile")


        # -------------------------------------------------
        # STUDENT DETAILS
        # -------------------------------------------------

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.metric(
                "Attendance",
                f"{student['Attendance']:.1f}%"
            )

        with a2:

            st.metric(
                "Study Hours",
                f"{student['Study_Hours']:.1f} hrs"
            )

        with a3:

            st.metric(
                "Assignment Score",
                f"{student['Assignment_Score']:.1f}"
            )

        with a4:

            st.metric(
                "Previous Semester",
                f"{student['Previous_Semester_Score']:.1f}"
            )


        st.divider()


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        st.subheader("🤖 Performance Prediction")

        input_data = pd.DataFrame(
            [[
                student["Attendance"],
                student["Study_Hours"],
                student["Assignment_Score"],
                student["Mid1_Score"],
                student["Mid2_Score"],
                student["Previous_Semester_Score"],
                student["Internal_Assessment"]
            ]],
            columns=features
        )

        predicted_score = model.predict(
            input_data
        )[0]

        predicted_score = np.clip(
            predicted_score,
            0,
            100
        )

        category = performance_category(
            predicted_score
        )


        prediction_col1, prediction_col2 = st.columns(2)

        with prediction_col1:

            st.metric(
                "Predicted Final Score",
                f"{predicted_score:.2f}/100"
            )

        with prediction_col2:

            if category == "High Performer":

                st.success(
                    "🟢 HIGH PERFORMER"
                )

            elif category == "Average Performer":

                st.warning(
                    "🟡 AVERAGE PERFORMER"
                )

            else:

                st.error(
                    "🔴 NEEDS ATTENTION"
                )


        # -------------------------------------------------
        # EARLY WARNING
        # -------------------------------------------------

        st.subheader("🚦 Early-Warning Monitor")

        risk1, risk2, risk3, risk4 = st.columns(4)


        with risk1:

            if student["Attendance"] < 75:

                st.error(
                    "🔴 Attendance\n\nNeeds attention"
                )

            else:

                st.success(
                    "🟢 Attendance\n\nGood"
                )


        with risk2:

            if student["Study_Hours"] < 3:

                st.warning(
                    "🟡 Study Hours\n\nMonitor"
                )

            else:

                st.success(
                    "🟢 Study Hours\n\nGood"
                )


        with risk3:

            if student["Assignment_Score"] < 60:

                st.error(
                    "🔴 Assignments\n\nNeeds attention"
                )

            else:

                st.success(
                    "🟢 Assignments\n\nGood"
                )


        with risk4:

            if student["Mid2_Score"] < student["Mid1_Score"]:

                st.warning(
                    "🟡 Exam Trend\n\nDeclining"
                )

            else:

                st.success(
                    "🟢 Exam Trend\n\nStable / Improving"
                )


        # -------------------------------------------------
        # KEY FACTORS
        # -------------------------------------------------

        st.subheader("🔎 Key Factors")

        factors = []


        if student["Attendance"] < 75:

            factors.append(
                "Low attendance"
            )


        if student["Study_Hours"] < 3:

            factors.append(
                "Low study hours"
            )


        if student["Assignment_Score"] < 60:

            factors.append(
                "Low assignment performance"
            )


        if student["Mid2_Score"] < student["Mid1_Score"]:

            factors.append(
                "Recent examination score declined"
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


        # -------------------------------------------------
        # ACADEMIC SUPPORT
        # -------------------------------------------------

        st.subheader(
            "💡 Suggested Academic Support"
        )

        suggestions = []


        if student["Attendance"] < 75:

            suggestions.append(
                "Monitor attendance regularly."
            )


        if student["Study_Hours"] < 3:

            suggestions.append(
                "Encourage a consistent study schedule."
            )


        if student["Assignment_Score"] < 60:

            suggestions.append(
                "Provide additional assignment support."
            )


        if student["Mid2_Score"] < student["Mid1_Score"]:

            suggestions.append(
                "Review recent examination performance."
            )


        if student["Previous_Semester_Score"] < 60:

            suggestions.append(
                "Provide additional academic mentoring."
            )


        if not suggestions:

            suggestions.append(
                "Continue current academic practices."
            )


        for suggestion in suggestions:

            st.write(
                "✅",
                suggestion
            )


    else:

        if search_id:

            st.warning(
                "Student not found. "
                "Please enter an ID such as "
                "S0001, S0500 or S1000."
            )

        else:

            st.info(
                "👆 Enter a Student ID or click "
                "🎬 Demo Student to begin."
            )


# =========================================================
# TAB 2 — ANALYTICS
# =========================================================

with tab2:

    st.header("📊 Academic Analytics Dashboard")

    st.write(
        "Explore relationships between academic factors "
        "and final performance."
    )


    # -----------------------------------------------------
    # PERFORMANCE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Performance Distribution"
    )

    category_data = pd.DataFrame({

        "Category": [
            "High Performer",
            "Average Performer",
            "Needs Attention"
        ],

        "Students": [
            high_count,
            average_count,
            attention_count
        ]
    })

    st.bar_chart(
        category_data.set_index("Category")
    )


    # -----------------------------------------------------
    # STUDY HOURS
    # -----------------------------------------------------

    st.subheader(
        "📚 Study Hours vs Final Score"
    )

    st.scatter_chart(
        df,
        x="Study_Hours",
        y="Final_Score"
    )


    # -----------------------------------------------------
    # ATTENDANCE
    # -----------------------------------------------------

    st.subheader(
        "🕐 Attendance vs Final Score"
    )

    st.scatter_chart(
        df,
        x="Attendance",
        y="Final_Score"
    )


    # -----------------------------------------------------
    # ASSIGNMENT
    # -----------------------------------------------------

    st.subheader(
        "📝 Assignment Score vs Final Score"
    )

    st.scatter_chart(
        df,
        x="Assignment_Score",
        y="Final_Score"
    )


    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.subheader(
        "🔗 Correlation Analysis"
    )

    correlation_columns = [
        "Attendance",
        "Study_Hours",
        "Assignment_Score",
        "Mid1_Score",
        "Mid2_Score",
        "Previous_Semester_Score",
        "Internal_Assessment",
        "Final_Score"
    ]

    correlation_matrix = df[
        correlation_columns
    ].corr()

    st.dataframe(
        correlation_matrix.round(2),
        use_container_width=True
    )


# =========================================================
# TAB 3 — PREDICTION MODEL
# =========================================================

with tab3:

    st.header("🤖 Prediction Engine")

    st.write(
        "The prototype uses Linear Regression to estimate "
        "expected final performance."
    )


    model_col1, model_col2, model_col3 = st.columns(3)


    with model_col1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

        st.caption(
            "Average absolute prediction error"
        )


    with model_col2:

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        st.caption(
            "Root mean squared prediction error"
        )


    with model_col3:

        st.metric(
            "R²",
            f"{r2:.2f}"
        )

        st.caption(
            "Variation explained by the model"
        )


    st.divider()


    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.subheader(
        "Features Used by the Model"
    )

    feature_display = pd.DataFrame({
        "Feature": features
    })

    st.dataframe(
        feature_display,
        hide_index=True,
        use_container_width=True
    )


    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader(
        "Actual vs Predicted Performance"
    )

    prediction_df = pd.DataFrame({

        "Actual Score": y_test.values,

        "Predicted Score": y_pred

    })

    st.scatter_chart(
        prediction_df,
        x="Actual Score",
        y="Predicted Score"
    )


# =========================================================
# TAB 4 — ABOUT SYSTEM
# =========================================================

with tab4:

    st.header("🏫 About the System")

    st.write(
        "This project is designed as an academic analytics "
        "and prediction layer that can complement an existing ERP."
    )


    # -----------------------------------------------------
    # SYSTEM WORKFLOW
    # -----------------------------------------------------

    st.subheader("🔄 System Workflow")

    workflow1, workflow2, workflow3 = st.columns(3)

    with workflow1:

        st.info(
            "📥 **Academic Data**\n\n"
            "Student academic records are collected "
            "for analysis."
        )

    with workflow2:

        st.info(
            "🧹 **Preprocessing**\n\n"
            "Missing values and duplicate records "
            "are handled."
        )

    with workflow3:

        st.info(
            "📊 **Analytics**\n\n"
            "Patterns and relationships in academic "
            "data are explored."
        )


    workflow4, workflow5, workflow6 = st.columns(3)

    with workflow4:

        st.info(
            "🤖 **Prediction**\n\n"
            "Machine learning estimates expected "
            "final performance."
        )

    with workflow5:

        st.warning(
            "🚦 **Early Warning**\n\n"
            "Potential academic risk factors "
            "are identified."
        )

    with workflow6:

        st.success(
            "💡 **Academic Support**\n\n"
            "Actionable academic-support suggestions "
            "are provided."
        )


    # -----------------------------------------------------
    # ERP COMPARISON
    # -----------------------------------------------------

    st.subheader(
        "🏫 Existing ERP vs Proposed Analytics Layer"
    )

    comparison = pd.DataFrame({

        "Existing ERP": [

            "Stores academic records",

            "Displays academic information",

            "Manages institutional processes"

        ],

        "Our Analytics Prototype": [

            "Analyzes academic patterns",

            "Predicts expected performance",

            "Identifies risk factors and support areas"

        ]
    })

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )


    # -----------------------------------------------------
    # FUTURE SCOPE
    # -----------------------------------------------------

    st.subheader(
        "🚀 Future Scope"
    )

    future_scope = [

        "Integration with authorized institutional ERP data",

        "More machine-learning algorithms",

        "Personalized academic interventions",

        "Longitudinal student performance tracking",

        "Explainable AI for predictions",

        "Privacy and role-based access controls"

    ]


    for item in future_scope:

        st.write(
            "🔹",
            item
        )


    # -----------------------------------------------------
    # PROTOTYPE NOTE
    # -----------------------------------------------------

    st.info(
        "Prototype note: The current system uses a synthetic "
        "academic dataset for demonstration. A real deployment "
        "would require authorized institutional data and "
        "appropriate privacy controls."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 Smart Student Performance Analytics and "
    "Prediction System | PBL Prototype"
)
