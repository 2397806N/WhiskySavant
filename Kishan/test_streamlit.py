import streamlit as st
import csv
import time
import keyboard
import os

st.write("1) Answer all the following questions by inputting your response via the keyboard")
st.write("2) Upon completion of your response, please submit it by pressing the 'Enter' key")
st.write("\n")

def data_to_collect(question, writer):
    st.write(question)
    start_time = time.time()
    first_letter_keyed = None
    last_letter_keyed = None
    No_of_backspace_used = 0
    input_text = ""

    key = f"user_input_{question.replace(' ', '_')}"  # Generate unique key for the text input widget

    text_input = st.empty()  # Create an empty text placeholder

    while True:
        event = keyboard.read_event(suppress=True)
        char = event.name

        if event.event_type == keyboard.KEY_DOWN:
            if char == "enter":
                end_time = time.time()

                # Calculate the time spent on the question
                time_spent_on_question = end_time - start_time

                # Calculate the average time spent per letter
                if len(input_text.replace(" ", "")) > 0:
                    time_spent_per_letter = time_spent_on_question / len(input_text.replace(" ", ""))
                else:
                    time_spent_per_letter = 0

                if first_letter_keyed is not None:
                    time_spent_before_first_letter_is_keyed = first_letter_keyed - start_time
                else:
                    time_spent_before_first_letter_is_keyed = 0

                if last_letter_keyed is not None:
                    time_to_submit_input_after_last_letter_keyed = end_time - last_letter_keyed
                else:
                    time_to_submit_input_after_last_letter_keyed = 0

                # Write data to CSV
                writer.writerow({
                    "Question": question,
                    "Input": input_text,
                    "Time spent on question": time_spent_on_question,
                    "Time spent per letter": time_spent_per_letter,
                    "Time spent before first letter is keyed": time_spent_before_first_letter_is_keyed,
                    "Time to submit input after last letter keyed": time_to_submit_input_after_last_letter_keyed,
                    "No. of backspace used": No_of_backspace_used
                })

                break
            elif char == "backspace":
                if input_text:
                    No_of_backspace_used += 1
                    input_text = input_text[:-1]
                    text_input.text(input_text)  # Update the text placeholder with the new input
            else:
                if first_letter_keyed is None:
                    first_letter_keyed = time.time()
                last_letter_keyed = time.time()
                if char == "space":
                    char = " "
                input_text += char  # Update the input text with the new character
                text_input.text(input_text)  # Update the text placeholder with the new input

def main():
    # Questions to ask
    questions = [
        "Name one country that you had ever travelled to",
        "What is one color that you like?",
        "Name one sport that you had ever done",
        "Which brand of handphone are you using now?",
        "Where do you stay in Singapore (name the town name)?",
        "What do people call you?"
    ]

    try:
        # Check if the file exists
        is_empty = not os.path.exists('survey_data.csv') or os.path.getsize('survey_data.csv') == 0
    except FileNotFoundError:
        is_empty = True

    with open('survey_data.csv', 'a', newline='') as csvfile:  # Use 'a' mode to append data
        fieldnames = ["Question", "Input", "Time spent on question", "Time spent per letter",
                      "Time spent before first letter is keyed", "Time to submit input after last letter keyed",
                      "No. of backspace used"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers only if the file is empty
        if is_empty:
            writer.writeheader()

        # Ask each question
        for q in questions:
            data_to_collect(q, writer)

    # Thank you message
    st.write("\n")
    st.write("Thank you for your responses!")

if __name__ == "__main__":
    main()
