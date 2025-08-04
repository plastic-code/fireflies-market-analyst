import streamlit as st
import re
import matplotlib.pyplot as plt

st.title("Market Size Visualizer (Manual Input Mode)")

# Text area for manual entry
summary_text = st.text_area("Paste Summary Text", height=300)

# Extract region and market size
def extract_market_data(text):
    pattern = r"-\s*(.+?):\s*\$?\s*([\d,.]+)\s*(B|M)?"
    results = []
    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            region = match.group(1).strip()
            amount = match.group(2).replace(",", "")
            scale = match.group(3)

            try:
                value = float(amount)
                if scale == "B":
                    value *= 1_000_000_000
                elif scale == "M":
                    value *= 1_000_000
                results.append((region, value))
            except ValueError:
                pass
    return results

# Parse and plot
data = extract_market_data(summary_text)
if not data:
    st.warning("⚠️ No valid $ market sizes found. Please include values like '$1.3B' or '$750M'.")
else:
    st.subheader("📊 Market Size by Region")
    regions, sizes = zip(*data)
    fig, ax = plt.subplots()
    ax.barh(regions, sizes)
    ax.set_xlabel("Market Size ($)")
    ax.invert_yaxis()
    st.pyplot(fig)
