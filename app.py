from pathlib import Path
import time

import streamlit as st

from main import (
    initialize_for_ui,
    index_all_documents,
    ask_chatbot,
)

from src.utils.file_manager import (
    ensure_directories,
    get_stored_documents,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Document AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>
    /* =========================
       DOCUMENT AI — UI SYSTEM
       ========================= */

    :root {
        --bg: #f6f7f9;
        --surface: #ffffff;
        --surface-2: #f9fafb;
        --border: #e6e8ec;
        --text: #17191d;
        --muted: #747a84;
        --muted-2: #9aa0aa;
        --accent: #111318;
        --success: #16a36a;
        --shadow: 0 1px 2px rgba(16, 24, 40, .04),
                   0 8px 30px rgba(16, 24, 40, .04);
    }

    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

    header, [data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 70% 0%, rgba(255,255,255,.95), transparent 32%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1120px !important;
        padding: 42px 34px 130px !important;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,.96) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 0 !important;
    }

    [data-testid="stSidebarContent"] {
        padding: 22px 16px !important;
    }

    .sidebar-brand {
        padding: 8px 8px 24px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: #111318;
        color: white;
        font-size: 14px;
        font-weight: 800;
        box-shadow: 0 5px 14px rgba(17,19,24,.15);
    }

    .sidebar-brand-title {
        font-size: 15px;
        font-weight: 750;
        letter-spacing: -.25px;
        color: var(--text);
    }

    .sidebar-brand-subtitle {
        margin-top: 2px;
        font-size: 11px;
        color: var(--muted);
    }

    .sidebar-section-title {
        margin: 20px 7px 9px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .9px;
        color: var(--muted-2);
    }

    /* ---------- Upload ---------- */

    .upload-box {
        padding: 15px;
        background: linear-gradient(180deg, #fafbfc, #f6f7f9);
        border: 1px dashed #d4d8df;
        border-radius: 13px;
        margin-bottom: 9px;
    }

    .upload-icon {
        width: 30px;
        height: 30px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: #eceef2;
        color: #30343b;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .upload-title {
        font-size: 13px;
        font-weight: 700;
        color: var(--text);
    }

    .upload-description {
        margin-top: 4px;
        font-size: 11px;
        line-height: 1.55;
        color: var(--muted);
    }

    [data-testid="stFileUploader"] {
        margin: 0 0 7px !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    [data-testid="stFileUploader"] button {
        width: 100% !important;
        min-height: 36px !important;
        border-radius: 9px !important;
        background: #111318 !important;
        color: white !important;
        border: 1px solid #111318 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
    }

    /* ---------- Sidebar controls ---------- */

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 36px;
        border-radius: 9px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: #30343b;
        font-size: 12px;
        font-weight: 650;
        transition: .15s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #c8ccd3;
        background: #f5f6f8;
    }

    /* ---------- Documents ---------- */

    .document-item {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 9px 8px;
        margin-bottom: 5px;
        border: 1px solid transparent;
        border-radius: 9px;
        transition: .15s ease;
    }

    .document-item:hover {
        background: #f7f8fa;
        border-color: var(--border);
    }

    .document-icon {
        width: 29px;
        height: 29px;
        flex: 0 0 29px;
        display: grid;
        place-items: center;
        border-radius: 7px;
        background: #f0f1f4;
        color: #606671;
        font-size: 8px;
        font-weight: 800;
        letter-spacing: .2px;
    }

    .document-name {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 11px;
        color: #454a53;
    }

    /* ---------- Main header ---------- */

    .app-header {
        margin-bottom: 24px;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 9px;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        color: #6f7580;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .15px;
        margin-bottom: 13px;
    }

    .eyebrow-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 0 3px rgba(22,163,106,.11);
    }

    .app-title {
        font-size: clamp(28px, 3vw, 38px);
        line-height: 1.08;
        font-weight: 780;
        letter-spacing: -1.5px;
        color: var(--text);
        margin: 0;
    }

    .app-subtitle {
        max-width: 650px;
        margin-top: 9px;
        font-size: 14px;
        line-height: 1.6;
        color: var(--muted);
    }

    /* ---------- Status ---------- */

    .status-bar {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 10px;
        border: 1px solid var(--border);
        border-radius: 9px;
        background: rgba(255,255,255,.78);
        color: #6d737d;
        font-size: 11px;
        margin-bottom: 24px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 0 3px rgba(22,163,106,.1);
    }

    /* ---------- Empty state ---------- */

    .empty-state {
        min-height: 440px;
        display: grid;
        place-items: center;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: rgba(255,255,255,.68);
        box-shadow: var(--shadow);
    }

    .empty-state-inner {
        width: min(520px, 90%);
        text-align: center;
        padding: 50px 20px;
    }

    .empty-state-icon {
        width: 58px;
        height: 58px;
        margin: 0 auto 18px;
        display: grid;
        place-items: center;
        border-radius: 16px;
        background: #111318;
        color: white;
        font-size: 16px;
        font-weight: 800;
        box-shadow: 0 10px 26px rgba(17,19,24,.13);
    }

    .empty-state-title {
        font-size: 21px;
        font-weight: 750;
        letter-spacing: -.45px;
        color: #25282e;
        margin-bottom: 8px;
    }

    .empty-state-text {
        max-width: 430px;
        margin: auto;
        font-size: 13px;
        line-height: 1.7;
        color: var(--muted);
    }

    /* ---------- Welcome workspace ---------- */

    .welcome-card {
        margin: 18px 0 28px;
        padding: 28px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(249,250,251,.88));
        box-shadow: var(--shadow);
    }

    .welcome-kicker {
        color: #8b919b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .welcome-title {
        color: var(--text);
        font-size: 20px;
        font-weight: 760;
        letter-spacing: -.45px;
    }

    .welcome-text {
        margin-top: 6px;
        max-width: 620px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
    }

    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 9px;
        margin-top: 22px;
    }

    .suggestion {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 13px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(255,255,255,.78);
    }

    .suggestion > span {
        width: 26px;
        height: 26px;
        flex: 0 0 26px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: #f0f1f4;
        color: #4e545e;
        font-size: 13px;
        font-weight: 700;
    }

    .suggestion b {
        display: block;
        color: #30343b;
        font-size: 11px;
        font-weight: 700;
    }

    .suggestion small {
        display: block;
        margin-top: 2px;
        color: #9298a2;
        font-size: 10px;
        line-height: 1.4;
    }

    /* ---------- Chat ---------- */

    [data-testid="stChatMessage"] {
        border: 0 !important;
        background: transparent !important;
        padding: 7px 0 !important;
        margin: 0 !important;
    }

    [data-testid="stChatMessage"] > div {
        max-width: 100% !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 14px !important;
        line-height: 1.72 !important;
        color: #30343b !important;
    }

    [data-testid="stChatMessage"] strong {
        color: #17191d !important;
    }

    [data-testid="stChatMessageAvatar"] {
        width: 32px !important;
        height: 32px !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
        max-width: min(76%, 700px) !important;
        margin-left: auto !important;
        padding: 11px 15px !important;
        border-radius: 14px 14px 4px 14px !important;
        background: #111318 !important;
        color: white !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) li {
        color: white !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
        max-width: min(88%, 820px) !important;
        padding: 15px 18px !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px 14px 14px 14px !important;
        background: rgba(255,255,255,.9) !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.03) !important;
    }

    /* ---------- Thinking ---------- */

    .thinking {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 9px 12px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: white;
        color: var(--muted);
    }

    .thinking-label {
        margin-right: 3px;
        font-size: 11px;
    }

    .thinking-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #777d87;
        animation: typing 1.2s infinite ease-in-out;
    }

    .thinking-dot:nth-child(2) { animation-delay: .15s; }
    .thinking-dot:nth-child(3) { animation-delay: .30s; }

    @keyframes typing {
        0%, 60%, 100% { opacity: .25; transform: translateY(0); }
        30% { opacity: 1; transform: translateY(-3px); }
    }

    /* ---------- Minimal capsule chat composer ---------- */

    [data-testid="stBottom"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    [data-testid="stBottomBlockContainer"] {
        max-width: 1120px !important;
        padding: 12px 34px 18px !important;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #d9dce2 !important;
        border-radius: 999px !important;
        background: #ffffff !important;
        box-shadow: 0 4px 18px rgba(16,24,40,.08) !important;
        overflow: hidden !important;
        transition: border-color .15s ease, box-shadow .15s ease !important;
    }

    [data-testid="stChatInput"]:hover {
        border-color: #c7cbd2 !important;
        box-shadow: 0 6px 22px rgba(16,24,40,.09) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #aeb3bc !important;
        box-shadow:
            0 6px 22px rgba(16,24,40,.09),
            0 0 0 3px rgba(17,19,24,.035) !important;
        transform: none !important;
    }

    [data-testid="stChatInput"] > div {
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        min-height: 46px !important;
        max-height: 120px !important;
        padding: 12px 58px 12px 20px !important;
        background: #ffffff !important;
        color: #111318 !important;
        -webkit-text-fill-color: #111318 !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        border: 0 !important;
        outline: none !important;
        box-shadow: none !important;
        resize: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9298a2 !important;
        -webkit-text-fill-color: #9298a2 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        min-height: 34px !important;
        right: 8px !important;
        bottom: 6px !important;
        border-radius: 50% !important;
        background: #111318 !important;
        color: #ffffff !important;
        border: 0 !important;
        box-shadow: none !important;
        transition: background .15s ease, transform .15s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #292c33 !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stChatInput"] button:active {
        transform: scale(.96) !important;
    }

    .composer-hint {
        display: none !important;
    }

    @media (max-width: 800px) {
        [data-testid="stBottomBlockContainer"] {
            padding: 10px 16px 14px !important;
        }

        [data-testid="stChatInput"] {
            border-radius: 999px !important;
        }

        [data-testid="stChatInput"] textarea {
            min-height: 44px !important;
            padding-left: 17px !important;
        }
    }





    /* ---------- Alerts ---------- */

    .stAlert {
        border-radius: 10px !important;
    }

    /* ---------- Scrollbars ---------- */

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d4d7dd; border-radius: 10px; }

    /* ---------- Responsive ---------- */

    @media (max-width: 800px) {
        [data-testid="stBottomBlockContainer"] {
            padding: 12px 16px 16px !important;
        }

        [data-testid="stChatInput"] {
            border-radius: 15px !important;
        }

        [data-testid="stChatInput"] textarea {
            min-height: 46px !important;
            padding-left: 14px !important;
        }

        .block-container {
            padding: 28px 18px 120px !important;
        }

        [data-testid="stBottomBlockContainer"] {
            padding-left: 18px !important;
            padding-right: 18px !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child,
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
            max-width: 92% !important;
        }
        .suggestion-grid { grid-template-columns: 1fr; }
        .welcome-card { padding: 22px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
# ============================================================
# INITIALIZE
# ============================================================
ensure_directories()
# ============================================================
# CHATBOT CACHE
# ============================================================
@st.cache_resource
def get_cached_chatbot():
    return initialize_for_ui()
# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="app-header">
      <div class="eyebrow"><span class="eyebrow-dot"></span>DOCUMENT INTELLIGENCE</div>
      <div class="app-title">Ask anything about your files.</div>
      <div class="app-subtitle">Your private workspace for searching, understanding, and extracting answers from uploaded documents.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-row">
                <div class="brand-mark">AI</div>
                <div>
                    <div class="sidebar-brand-title">Document AI</div>
                    <div class="sidebar-brand-subtitle">Knowledge workspace</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # --------------------------------------------------------
    # Upload section
    # --------------------------------------------------------
    st.markdown(
        '<div class="sidebar-section-title">Add documents</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="upload-box">
            <div class="upload-icon">↑</div>
            <div class="upload-title">Add a document</div>
            <div class="upload-description">
                PDF, TXT, DOCX, DOC, Markdown or CSV
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload",
        type=[
            "pdf",
            "txt",
            "docx",
            "doc",
            "md",
            "csv",
        ],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button(
            "Upload and process",
            use_container_width=True,
        ):
            try:
                documents_dir = Path("documents")
                documents_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                destination = (
                    documents_dir
                    / uploaded_file.name
                )
                with open(
                    destination,
                    "wb",
                ) as file:
                    file.write(
                        uploaded_file.getbuffer()
                    )
                with st.spinner(
                    "Processing document..."
                ):
                    total = index_all_documents(
                        show_progress=False
                    )
                get_cached_chatbot.clear()
                st.success(
                    f"{uploaded_file.name} is ready."
                )
                time.sleep(0.7)
                st.rerun()
            except Exception as exc:
                st.error(
                    f"Upload failed: {exc}"
                )
    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------
    st.markdown(
        '<div class="sidebar-section-title">Documents</div>',
        unsafe_allow_html=True,
    )
    try:
        documents = get_stored_documents()
        if documents:
            for document in documents:
                extension = (
                    Path(document.name)
                    .suffix
                    .replace(".", "")
                    .upper()
                )
                if not extension:
                    extension = "DOC"
                st.markdown(
                    f"""
                    <div class="document-item">
                        <div class="document-icon">
                            {extension[:4]}
                        </div>
                        <div class="document-name"
                             title="{document.name}">
                            {document.name}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption(
                "No documents uploaded."
            )
    except Exception as exc:
        st.warning(
            f"Could not load documents: {exc}"
        )
    # --------------------------------------------------------
    # Bottom controls
    # --------------------------------------------------------
    st.markdown(
        '<div style="height: 18px;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-section-title">Conversation</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()
# ============================================================
# DOCUMENT CHECK
# ============================================================
documents = get_stored_documents()
if not documents:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-inner">
                <div class="empty-state-icon">AI</div>
                <div class="empty-state-title">Your workspace is ready</div>
                <div class="empty-state-text">
                    Start by uploading a document from the sidebar.
                    Once it is processed, you can ask questions, find
                    information, and explore your files conversationally.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
# ============================================================
# CHATBOT INITIALIZATION
# ============================================================
try:
    chatbot = get_cached_chatbot()
except Exception as exc:
    st.error(
        f"Could not initialize the AI assistant:\n\n{exc}"
    )
    st.stop()
# ============================================================
# STATUS
# ============================================================
st.markdown(
    """
    <div class="status-bar">
        <span class="status-dot"></span>
        <span>
            AI ready · indexed documents available
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
          <div class="welcome-copy">
            <div class="welcome-kicker">READY TO SEARCH</div>
            <div class="welcome-title">What would you like to find?</div>
            <div class="welcome-text">Ask a question below. The assistant will use your indexed documents to build the answer.</div>
          </div>
          <div class="suggestion-grid">
            <div class="suggestion"><span>⌕</span><div><b>Find information</b><small>Locate a specific fact or section</small></div></div>
            <div class="suggestion"><span>≡</span><div><b>Summarize a document</b><small>Turn long content into key points</small></div></div>
            <div class="suggestion"><span>↗</span><div><b>Compare details</b><small>Connect information across files</small></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
# ============================================================
# CHAT INPUT
# ============================================================
question = st.chat_input(
    "Ask anything about your documents..."
)
# ============================================================
# HANDLE QUESTION
# ============================================================
if question:
    question = question.strip()
    if not question:
        st.stop()
    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )
    with st.chat_message("user"):
        st.markdown(question)
    # --------------------------------------------------------
    # ASSISTANT MESSAGE
    # --------------------------------------------------------
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            """
            <div class="thinking">
                <span class="thinking-label">
                    Searching documents
                </span>
                <span class="thinking-dot"></span>
                <span class="thinking-dot"></span>
                <span class="thinking-dot"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            # -----------------------------------------------
            # YOUR EXISTING RAG SYSTEM
            # -----------------------------------------------
            answer = ask_chatbot(
                chatbot,
                question,
            )
            if answer is None:
                answer = "No answer was found."
            answer = str(answer)
            # -----------------------------------------------
            # Remove thinking indicator
            # -----------------------------------------------
            thinking_placeholder.empty()
            # -----------------------------------------------
            # Typing animation
            # -----------------------------------------------
            response_placeholder = st.empty()
            displayed_text = ""
            # Character-based animation
            #
            # Increase to 0.01 for faster
            # Decrease to 0.02 for slower
            for character in answer:
                displayed_text += character
                response_placeholder.markdown(
                    displayed_text
                )
                time.sleep(0.008)
        except Exception as exc:
            thinking_placeholder.empty()
            answer = (
                "An error occurred while generating "
                "the answer:\n\n"
                f"{exc}"
            )
            st.error(answer)
    # --------------------------------------------------------
    # SAVE RESPONSE
    # --------------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )