import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 20px;
    margin-top: 0px;
    opacity: 0.8;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
}

.info-card {
    padding: 22px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.score-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
}

.score-number {
    font-size: 42px;
    font-weight: 800;
}

.small-label {
    font-size: 14px;
    opacity: 0.75;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("student_performance_cleaned.csv")


# =========================================================
# MACHINE LEARNING
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
        .str.upper()
        .str.strip()
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

    # No arrows — clean pipeline
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
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🎓 Smart Student Performance Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict • Identify • Improve'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "A data-driven academic analytics and early-warning "
    "system for student performance monitoring."
)

st.divider()


# =========================================================
# KPI CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "👨‍🎓 Total Students",
        f"{len(df):,}"
    )

with c2:

    st.metric(
        "📊 Average Score",
        f"{df['Final_Score'].mean():.2f}"
    )

with c3:

    st.metric(
        "🤖 Model R²",
        f"{r2:.2f}"
    )

with c4:

    st.metric(
        "📉 Prediction Error",
        f"{mae:.2f}"
    )


# =========================================================
# PERFORMANCE OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📊 Academic Performance Overview'
    '</div>',
    unsafe_allow_html=True
)

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

    st.markdown(
        '<div class="section-title">'
        '🔎 Student Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Search any student using the Student ID from the dataset."
    )

    search_col, demo_col = st.columns([4, 1])

    with search_col:

        search_id = st.text_input(
            "Student ID",
            placeholder="Example: S0500",
            key="student_search"
        )

    with demo_col:

        st.write("")

        demo_clicked = st.button(
            "🎬 Demo Student",
            use_container_width=True
        )

    # -----------------------------------------------------
    # DEMO STUDENT
    # -----------------------------------------------------

    if demo_clicked:

        st.session_state["demo_student"] = "S0500"

        st.rerun()


    if (
        "demo_student" in st.session_state
        and not search_id
    ):

        search_id = st.session_state["demo_student"]


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

        st.markdown(
            '<div class="section-title">'
            '👤 Student Profile'
            '</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # STUDENT METRICS
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
                "Assignment",
                f"{student['Assignment_Score']:.1f}"
            )

        with a4:

            st.metric(
                "Previous Semester",
                f"{student['Previous_Semester_Score']:.1f}"
            )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

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


        st.markdown(
            '<div class="section-title">'
            '🤖 Performance Prediction'
            '</div>',
            unsafe_allow_html=True
        )


        r1, r2_col = st.columns([1, 2])

        with r1:

            st.markdown(
                f"""
                <div class="score-card">

                    <div class="small-label">
                        Predicted Final Score
                    </div>

                    <div class="score-number">
                        {predicted_score:.2f}
                    </div>

                    <div>
                        out of 100
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with r2_col:

            if category == "High Performer":

                st.success(
                    "🟢 HIGH PERFORMER\n\n"
                    "The predicted performance is in the "
                    "high-performance range."
                )

            elif category == "Average Performer":

                st.warning(
                    "🟡 AVERAGE PERFORMER\n\n"
                    "The student may benefit from continued "
                    "academic monitoring."
                )

            else:

                st.error(
                    "🔴 NEEDS ATTENTION\n\n"
                    "The student may require additional "
                    "academic support."
                )


        # -------------------------------------------------
        # EARLY WARNING
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🚦 Early-Warning Monitor'
            '</div>',
            unsafe_allow_html=True
        )

        risk_items = []


        if student["Attendance"] < 75:

            risk_items.append(
                ("🔴", "Attendance", "Needs attention")
            )

        else:

            risk_items.append(
                ("🟢", "Attendance", "Good")
            )


        if student["Study_Hours"] < 3:

            risk_items.append(
                ("🟡", "Study Hours", "Monitor")
            )

        else:

            risk_items.append(
                ("🟢", "Study Hours", "Good")
            )


        if student["Assignment_Score"] < 60:

            risk_items.append(
                ("🔴", "Assignments", "Needs attention")
            )

        else:

            risk_items.append(
                ("🟢", "Assignments", "Good")
            )


        if student["Mid2_Score"] < student["Mid1_Score"]:

            risk_items.append(
                ("🟡", "Exam Trend", "Declining")
            )

        else:

            risk_items.append(
                ("🟢", "Exam Trend", "Stable / Improving")
            )


        risk_cols = st.columns(4)


        for i, item in enumerate(risk_items):

            with risk_cols[i]:

                icon, label, status = item

                st.markdown(
                    f"""
                    <div class="info-card">

                        <b>{icon} {label}</b>
                        <br>

                        {status}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # -------------------------------------------------
        # KEY FACTORS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🔎 Key Factors'
            '</div>',
            unsafe_allow_html=True
        )

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

        st.markdown(
            '<div class="section-title">'
            '💡 Suggested Academic Support'
            '</div>',
            unsafe_allow_html=True
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
                "Student not found. Please enter an ID "
                "such as S0001, S0500 or S1000."
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

    st.markdown(
        '<div class="section-title">'
        '📊 Academic Analytics Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

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

    st.markdown(
        '<div class="section-title">'
        '🤖 Prediction Engine'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The prototype uses Linear Regression to estimate "
        "expected final performance."
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

        st.caption(
            "Average absolute prediction error"
        )


    with m2:

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        st.caption(
            "Root mean squared prediction error"
        )


    with m3:

        st.metric(
            "R²",
            f"{r2:.2f}"
        )

        st.caption(
            "Variation explained by the model"
        )


    st.divider()


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

    st.markdown(
        '<div class="section-title">'
        '🏫 About the System'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This project is designed as an academic analytics "
        "and prediction layer that can complement an existing ERP."
    )


    # -----------------------------------------------------
    # WORKFLOW
    # -----------------------------------------------------

    st.subheader(
        "🔄 System Workflow"
    )

    workflow_cols = st.columns(6)

    workflow = [
        ("📥", "Academic Data"),
        ("🧹", "Preprocessing"),
        ("📊", "Analytics"),
        ("🤖", "Prediction"),
        ("🚦", "Early Warning"),
        ("💡", "Academic Support")
    ]


    for col, item in zip(
        workflow_cols,
        workflow
    ):

        with col:

            icon, text = item

            st.markdown(
                f"""
                <div class="info-card"
                     style="text-align:center">

                    <div style="font-size:28px">
                        {icon}
                    </div>

                    <b>{text}</b>

                </div>
                """,
                unsafe_allow_html=True
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
