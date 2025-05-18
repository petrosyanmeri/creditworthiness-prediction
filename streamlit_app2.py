import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Վարկունակության գնահատում", layout="centered")

st.markdown("<h1 style='text-align: center;'> Վարկունակության գնահատում</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'> Վարկունակության կանխատեսում AI մոդելի միջոցով։</p>", unsafe_allow_html=True)

# Load model
MODEL_PATH = Path(__file__).resolve().parent / "creditworthiness.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# Mapping Armenian to English for model
marital_status_map = {
    "Ամուսնացած": "Married",
    "Միայնակ": "Single",
    "Բաժանված": "Divorced",
    "Ամուրի": "Widowed"
}
employment_status_map = {
    "Աշխատում եք": "Employed",
    "Չեք աշխատում": "Unemployed",
    "Ինքնազբաղված": "Self-Employed",
    "Թոշակառու": "Retired"
}
education_map = {
    "Միջնակարգ": "High School",
    "Բակալավր": "Bachelor",
    "Մագիստրոս": "Master",
    "Դոկտոր": "PhD",
    "Չունեք": "None"
}
collateral_map = {
    "Այո": "Yes",
    "Ոչ": "No"
}
loan_purpose_map = {
    "Հիփոթեք": "Mortgage",
    "Ավտովարկ": "Auto Loan",
    "Անձնական վարկ": "Personal Loan",
    "Բիզնես վարկ": "Business Loan",
    "Ուսանողական վարկ": "Education Loan"
}


# Enhanced FICO-like calculation using official weights
def calc_fico(income: int, debt: int, age: int, loan_amount: int):
    base_score = 300

    # 1. Payment History surrogate (35%) → debt-to-income ratio
    dti = debt / income if income > 0 else 1
    payment_history_score = (1 - min(dti, 1)) * 35

    # 2. Amounts Owed (30%) → level of total debt
    amounts_owed_score = (1 - min(debt / 200000, 1)) * 30

    # 3. Length of Credit History (15%) → proxy: borrower's age
    length_history_score = min(age / 100, 1) * 15

    # 4. New Credit (10%) → size of new loan
    new_credit_penalty = min(loan_amount / 50000, 1) * 10
    new_credit_score = 10 - new_credit_penalty

    # 5. Credit Mix (10%) → fixed small bonus due to lack of data
    credit_mix_score = 8

    total_score = base_score \
        + payment_history_score \
        + amounts_owed_score \
        + length_history_score \
        + new_credit_score \
        + credit_mix_score

    # Ensure within 300-850
    return int(np.clip(total_score, 300, 850))


def encode(val: str, categories: list[str]) -> int:
    le = LabelEncoder().fit(categories)
    return int(le.transform([val])[0])

# App
st.write("Խնդրում ենք լրացնել բոլոր դաշտերը հաշվարկի համար։")

with st.form("credit_form"):
    # --- Անձնական տվյալներ ---
    st.subheader("👤 Անձնական Տվյալներ")
    full_name = st.text_input("Անուն Ազգանուն")
    error_full_name = st.empty()

    age = st.selectbox("Տարիք", ["— Ընտրեք տարիքը —"] + list(range(18, 101)))
    error_age = st.empty()

    monthly_income = st.text_input("Ամսական եկամուտ (AMD)")
    error_monthly_income = st.empty()

    debt = st.text_input("Ընդհանուր պարտք (AMD)")
    error_debt = st.empty()

    loan_amount = st.text_input("Վարկի գումար (AMD)")
    error_loan_amount = st.empty()

    # --- Ֆինանսական ցուցանիշներ ---
    st.subheader("🏦 Ֆինանսական Ցուցանիշներ")
    num_bank_accounts = st.text_input("Բանկային հաշիվների քանակ")
    error_num_bank_accounts = st.empty()

    num_credit_cards = st.text_input("Կրեդիտ քարտերի քանակ")
    error_num_credit_cards = st.empty()

    num_of_loans = st.text_input("Վարկերի քանակ")
    error_num_of_loans = st.empty()

    num_delayed_payments = st.text_input("Ուշացրած վճարումների քանակ")
    error_num_delayed_payments = st.empty()

    credit_history_age = st.text_input("Վարկի ժամկետ (ամիսներով)")
    error_credit_history_age = st.empty()

    # --- Վարկային և աշխատանքային տվյալներ ---
    st.subheader("📌 Վարկային և Աշխատանքային Տվյալներ")
    marital_status_arm = st.selectbox("Ամուսնական կարգավիճակ", ["— ԸՆՏՐԵԼ —"] + list(marital_status_map.keys()))
    error_marital_status = st.empty()

    employment_status_arm = st.selectbox("Աշխատանքային կարգավիճակ", ["— ԸՆՏՐԵԼ —"] + list(employment_status_map.keys()))
    error_employment_status = st.empty()

    education_level_arm = st.selectbox("Կրթության մակարդակ", ["— ԸՆՏՐԵԼ —"] + list(education_map.keys()))
    error_education_level = st.empty()

    collateral_arm = st.selectbox("Գրավ կա՞", ["— ԸՆՏՐԵԼ —"] + list(collateral_map.keys()))
    error_collateral = st.empty()

    loan_purpose_arm = st.selectbox("Վարկի նպատակը", ["— ԸՆՏՐԵԼ —"] + list(loan_purpose_map.keys()))
    error_loan_purpose = st.empty()

    # Buttons
    calculate_fico = st.form_submit_button("📐 Հաշվել FICO սքոր")
    predict_clicked = st.form_submit_button("📈 Հաշվել վարկունակությունը")

    # General error below buttons
    error_general = st.empty()

    # Validation function
    def validate_all(require_full_model: bool):
        ok = True
        # Clear previous errors
        for ph in [error_full_name, error_age, error_monthly_income, error_debt, error_loan_amount,
                   error_num_bank_accounts, error_num_credit_cards, error_num_of_loans,
                   error_num_delayed_payments, error_credit_history_age,
                   error_marital_status, error_employment_status, error_education_level,
                   error_collateral, error_loan_purpose, error_general]:
            ph.empty()

        # Full model: require name
        if require_full_model and not full_name.strip():
            error_full_name.error("Փոփոխություն է պահանջվում։")
            ok = False
        # Age
        if age == "— Ընտրեք տարիքը —":
            error_age.error("Ընտրել տարիքը։")
            ok = False

        # Numeric fields
        numeric_fields = [
            (monthly_income, error_monthly_income), (debt, error_debt), (loan_amount, error_loan_amount),
            (num_bank_accounts, error_num_bank_accounts), (num_credit_cards, error_num_credit_cards),
            (num_of_loans, error_num_of_loans), (num_delayed_payments, error_num_delayed_payments),
            (credit_history_age, error_credit_history_age)
        ]
        for val, ph in numeric_fields:
            if not val.strip():
                ph.error("Պետք է լրացվնել")
                ok = False
            else:
                try:
                    num = float(val)
                    if num < 0:
                        ph.error("Չի կարող բացասական լինել։")
                        ok = False
                except:
                    ph.error("Պետք է թվային լինի։")
                    ok = False

        # Select fields for full model
        if require_full_model:
            if marital_status_arm == "— ԸՆՏՐԵԼ —":
                error_marital_status.error("Ընտրել արժեք։")
                ok = False
            if employment_status_arm == "— ԸՆՏՐԵԼ —":
                error_employment_status.error("Ընտրել արժեք։")
                ok = False
            if education_level_arm == "— ԸՆՏՐԵԼ —":
                error_education_level.error("Ընտրել արժեք։")
                ok = False
            if collateral_arm == "— ԸՆՏՐԵԼ —":
                error_collateral.error("Ընտրել արժեք։")
                ok = False
            if loan_purpose_arm == "— ԸՆՏՐԵԼ —":
                error_loan_purpose.error("Ընտրել արժեք։")
                ok = False

        if not ok:
            error_general.error("Պետք է լրացնել բոլոր դաշտերը։")
        return ok

    # Actions
    if calculate_fico:
        if validate_all(require_full_model=False):
            monthly_income, debt, loan_amount = float(monthly_income), float(debt), float(loan_amount)
            fico_score = calc_fico(monthly_income, debt, age, loan_amount)
            st.markdown(f"""
                <div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px; margin-top: 10px; text-align: center;">
                    <h3>📌 Ձեր հաշվարկված FICO սկորն է՝ <span style="color: #007acc;">{fico_score}</span></h3>
                </div>
            """, unsafe_allow_html=True)

    if predict_clicked:
        if validate_all(require_full_model=True):
            # Parse numeric fields
            monthly_income = float(monthly_income)
            debt = float(debt)
            loan_amount = float(loan_amount)
            num_bank_accounts = int(num_bank_accounts)
            num_credit_cards = int(num_credit_cards)
            num_of_loans = int(num_of_loans)
            num_delayed_payments = int(num_delayed_payments)
            credit_history_age = int(credit_history_age)

            # Translate to English and encode
            marital_status_encoded = encode(marital_status_map[marital_status_arm], list(marital_status_map.values()))
            employment_status_encoded = encode(employment_status_map[employment_status_arm], list(employment_status_map.values()))
            education_level_encoded = encode(education_map[education_level_arm], list(education_map.values()))
            collateral_encoded = encode(collateral_map[collateral_arm], list(collateral_map.values()))
            loan_purpose_encoded = encode(loan_purpose_map[loan_purpose_arm], list(loan_purpose_map.values()))

            # Calculate FICO for model input and DTI
            fico_score = calc_fico(monthly_income, debt, age, loan_amount)
            debt_to_income_ratio = debt / monthly_income if monthly_income > 0 else 0

            # Prepare DataFrame for prediction
            input_dict = {
                "Age": age,
                "Monthly_Income": monthly_income,
                "Debt": debt,
                "Loan_Amount": loan_amount,
                "Marital_Status": marital_status_encoded,
                "Employment_Status": employment_status_encoded,
                "Education_Level": education_level_encoded,
                "Collateral": collateral_encoded,
                "Loan_Purpose": loan_purpose_encoded,
                "Num_Bank_Accounts": num_bank_accounts,
                "Num_Credit_Cards": num_credit_cards,
                "Num_of_Loans": num_of_loans,
                "Num_Delayed_Payments": num_delayed_payments,
                "Credit_History_Age": credit_history_age,
                "FICO_Score": fico_score,
                "Debt_to_Income_Ratio": debt_to_income_ratio
            }
            input_df = pd.DataFrame([input_dict])

            # Predict
            proba = model.predict_proba(input_df)[0]
            prediction = int(np.argmax(proba))
            confidence = round(proba[prediction] * 100, 2)

            cmap = {0: ("❌ Վատ", "#ffcccc"), 1: ("⚠️ Միջին", "#fff0b3"), 2: ("✅ Լավ", "#ccffcc")}
            result_label, color = cmap.get(prediction, ("Անհայտ", "#eeeeee"))

            st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 15px; margin-top: 20px;">
                    <h2 style="text-align: center;">Հարգելի <span style="color:#3366cc;">{full_name}</span>,</h2>
                    <h3 style="text-align: center;">Դուք պատկանում եք Վարկունակության <strong>{result_label}</strong> դասին։</h3>
                    <p style="text-align: center; font-size: 18px;">📈 Կանխատեսման վստահություն՝ <strong>{confidence}%</strong></p>
                </div>
            """, unsafe_allow_html=True)
