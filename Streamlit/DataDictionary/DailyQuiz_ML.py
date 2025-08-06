import streamlit as st
import pandas as pd
import datetime
import os

# --- 1. Load your data
@st.cache_data
def load_data():
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv'
    return pd.read_csv(url)

df = load_data()

# --- 2. Initialize state
if 'word_list' not in st.session_state:
    st.session_state.word_list = df.sample(frac=1).reset_index(drop=True)
    st.session_state.current_index = 0
    st.session_state.answers = []

# --- 3. Quiz flow
if st.session_state.current_index < len(st.session_state.word_list):
    current_word = st.session_state.word_list.iloc[st.session_state.current_index]
    word_text = str(current_word['Word']).strip()
    st.header(f"Word {st.session_state.current_index + 1} of {len(st.session_state.word_list)}")
    st.write(f"**Word:** {word_text}")

    # --- Definition fallback
    definition = str(current_word['Definition']).strip()
    if definition.lower() == 'nan' or definition == '':
        definition = "No definition available."

    # --- PERSISTENT toggle for Definition
    show_def = st.toggle(
        "Show Definition",
        value=False,
        key=f"show_def_{st.session_state.current_index}"
    )
    if show_def:
        st.info(f"**Definition:** {definition}")

    # --- PERSISTENT toggle for Details (all other columns)
    show_details = st.toggle(
        "Show Details",
        value=False,
        key=f"show_details_{st.session_state.current_index}"
    )
    if show_details:
        other_cols = current_word.drop(labels=['Word', 'Definition'])
        st.write("📄 **Additional Details:**")
        st.dataframe(pd.DataFrame(other_cols).rename(columns={0: 'Value'}))

    # --- Use a form to fully isolate answer + Next Word
    with st.form(key=f"form_{st.session_state.current_index}"):
        answer = st.radio(
            "How did you do?",
            options=['Correct', 'Pass', 'Incorrect'],
            index=None
        )
        submitted = st.form_submit_button("Next Word")

        if submitted:
            if answer is None:
                st.warning("⚠️ Please select an option before continuing.")
            else:
                st.session_state.answers.append({
                    'Word': word_text,
                    'Definition': definition,
                    'Answer': answer
                })
                st.session_state.current_index += 1

    # --- End Quiz button
    if st.button("End Quiz Now"):
        st.session_state.current_index = len(st.session_state.word_list)

# --- 4. Final results & download
if st.session_state.current_index >= len(st.session_state.word_list):
    num_attempted = len(st.session_state.answers)

    if num_attempted < 10:
        st.warning(f"You must complete at least 10 iterations. You've only done {num_attempted}.")
        if st.button("Continue Quiz"):
            st.session_state.current_index = num_attempted
    else:
        st.success(f"✅ You completed {num_attempted} iterations!")

        result_df = pd.DataFrame(st.session_state.answers)
        st.dataframe(result_df)

        num_correct = sum(1 for a in st.session_state.answers if a['Answer'] == 'Correct')
        num_pass = sum(1 for a in st.session_state.answers if a['Answer'] == 'Pass')
        num_incorrect = sum(1 for a in st.session_state.answers if a['Answer'] == 'Incorrect')

        st.write(f"**✅ Correct:** {num_correct}")
        st.write(f"**⏭️ Pass:** {num_pass}")
        st.write(f"**❌ Incorrect:** {num_incorrect}")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f'/Documents/Python/Github_Repo/Streamlit/ScoreFolder/quiz_results_{timestamp}.csv',
            mime='text/csv'
        )
        
        folder_path = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/ScoreFolder'
        os.makedirs(folder_path, exist_ok=True)  # create if it doesn't exist

        file_name = f'quiz_results_{timestamp}.csv'
        file_path = os.path.join(folder_path, file_name)

        result_df.to_csv(file_path, index=False)
        st.success(f"✅ Results saved locally at: `{file_path}`")

        # --- 2️⃣ Also provide a download button for browser download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=file_name,  # just the name, no path
            mime='text/csv'
        )