import streamlit as st
import re
import matplotlib.pyplot as plt

st.title("Market Summary to Graph Visualizer")

uploaded_file = st.file_uploader("Upload a Market Summary (.txt)", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    st.subheader("Raw Summary")
    st.text_area("Summary Text", text, height=200)

    # Look for patterns like: Saudi Arabia ... SAR 70B
    matches = re.findall(r"(Saudi Arabia|UAE|Egypt|Qatar|Kuwait|Bahrain).*?SAR\s(\d+)B", text)

    if matches:
        labels = [match[0] for match in matches]
        values = [int(match[1]) for match in matches]

        st.subheader("Extracted Market Sizes")
        for label, value in zip(labels, values):
            st.write(f"{label}: SAR {value}B")

        st.subheader("Market Size Comparison")
        fig, ax = plt.subplots()
        ax.bar(labels, values, color="skyblue")
        ax.set_ylabel("SAR (Billion)")
        ax.set_title("Market Sizes by Country")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No valid SAR market sizes found. Check the format.")
