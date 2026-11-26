"""按结果文件的修改时间缓存看板使用的逐题记录。"""

import json

import streamlit as st


@st.cache_data(show_spinner=False)
def load_details(path, modified_ns):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]
