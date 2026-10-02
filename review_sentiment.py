"""
RESTAURANT / RECIPE REVIEW SENTIMENT ANALYTICS

The models train directly from Recipe_Dataset.csv when no labelled file is
uploaded. SMOTE balances only the training split; the held-out test split remains unchanged.
Optional uploaded training CSV requirements: `text` and `stars` columns.

Rating guide:
4-5 stars = positive | 3 stars = neutral | 1-2 stars = negative
0 stars = excluded
"""


# 1. IMPORT PACKAGES

from io import BytesIO                 # Lets us read uploaded files in memory.
from pathlib import Path               # Locates Recipe_Dataset.csv beside this script.
import re                              # Provides regular expressions.
import textwrap                        # Wraps long text in PDF reports.

import matplotlib.pyplot as plt        # Creates evaluation figures.
from matplotlib.backends.backend_pdf import PdfPages  # Creates PDF downloads.
import numpy as np                     # Supports numerical operations.
import pandas as pd                    # Reads and processes CSV data.
from pypdf import PdfReader            # Extracts text from uploaded PDFs.
import streamlit as st                 # Creates the user interface.
from imblearn.over_sampling import SMOTE  # Balances the training examples.

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC



# 2. PAGE AND PROJECT SETTINGS

st.set_page_config(
    page_title='Restaurant Review Analytics',
    page_icon='🍲',
    layout='wide',
)

RANDOM_STATE = 42                     # Makes results repeatable.
TEST_SIZE = 0.20                      # Uses 20% of uploaded data for testing.
CLASS_NAMES = ['negative', 'neutral', 'positive']

# Keep the training CSV beside this Python file. Models are built in memory.
TRAINING_CSV_FILE = Path(__file__).with_name('Recipe_Dataset.csv')



# 3. SESSION STATE

# Streamlit reruns after each click. These variables preserve the user's uploads.
st.session_state.setdefault('logged_in', False)
st.session_state.setdefault('raw_data', None)
st.session_state.setdefault('clean_data', None)
st.session_state.setdefault('cleaning_audit', None)
st.session_state.setdefault('training_filename', None)
st.session_state.setdefault('prediction_results', None)
st.session_state.setdefault('prediction_source', None)
st.session_state.setdefault('landing_sentiment', None)
st.session_state.setdefault('landing_model', None)
# Remember whether the latest prediction came from typed text or a file.
st.session_state.setdefault('prediction_input_kind', None)



# 4. SIMPLE PRESENTATION LOGIN

def login_page():
    """Show a basic classroom login; this is not production security."""
    st.title('🍲 Restaurant / recipe review analytics', text_alignment='center')
    st.caption(
        'Group 4 - TF-IDF - Support Vector Machine - Random Forest',
        text_alignment='center',
    )

    with st.container(horizontal_alignment='center'):
        with st.form('login_form', width=420):
            st.subheader('Welcome', text_alignment='center')
            username = st.text_input('Username', icon=':material/person:')
            password = st.text_input(
                'Password', type='password', icon=':material/lock:'
            )
            submitted = st.form_submit_button(
                'Log in', type='primary', icon=':material/login:', width='stretch'
            )

        st.caption(
            'Demo access: username `group4`, password `recipe2026`.',
            text_alignment='center',
        )

    if submitted:
        if username.strip().lower() == 'group4' and password == 'recipe2026':
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error('Incorrect username or password.')


if not st.session_state.logged_in:
    login_page()
    st.stop()



# 5. TEXT CLEANING

def clean_text(text):
    """Clean one review using understandable regex operations."""
    text = str(text).lower()                          # Convert to lowercase.
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Remove URLs.
    text = re.sub(r'<[^>]+>', ' ', text)             # Remove HTML tags.
    text = re.sub(r'[^a-z\s]', ' ', text)            # Keep letters/spaces.
    text = re.sub(r'\s+', ' ', text).strip()         # Normalise spaces.
    return text



# 6. PROCESS A TRAINING CSV UPLOADED BY THE USER

@st.cache_data(show_spinner=False)
def process_training_csv(file_bytes):
    """Validate, clean and label an uploaded training CSV."""
    raw = pd.read_csv(BytesIO(file_bytes))            # Read uploaded bytes.

    required = {'text', 'stars'}                      # Required columns.
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError('Missing column(s): ' + ', '.join(sorted(missing)))

    data = raw.copy()                                 # Preserve raw corpus.
    audit = [['Raw uploaded dataset', len(data)]]

    data['stars'] = pd.to_numeric(data['stars'], errors='coerce')
    data = data.dropna(subset=['stars', 'text'])      # Remove missing rows.
    data['text'] = data['text'].astype(str).str.strip()
    data = data[data['text'] != '']                   # Remove blank text.
    audit.append(['After missing/blank removal', len(data)])

    data = data.drop_duplicates(subset=['text'])      # Remove repeated text.
    if 'comment_id' in data.columns:
        data = data.drop_duplicates(subset=['comment_id'])
    audit.append(['After duplicate removal', len(data)])

    data = data[data['stars'].between(1, 5)].copy()   # Exclude 0 stars.
    audit.append(['After excluding 0/invalid stars', len(data)])

    data['Clean_Text'] = data['text'].apply(clean_text)
    data['Word_Count'] = data['Clean_Text'].str.split().str.len()
    data = data[data['Word_Count'] > 0].copy()

    # Apply the exact rating guide supplied by the user.
    data['Class'] = np.select(
        [data['stars'].between(1, 2), data['stars'] == 3,
         data['stars'].between(4, 5)],
        ['negative', 'neutral', 'positive'],
        default='excluded',
    )
    data = data[data['Class'].isin(CLASS_NAMES)].copy()
    audit.append(['Final usable dataset', len(data)])

    # All three classes are necessary for this multiclass project.
    missing_classes = set(CLASS_NAMES).difference(data['Class'].unique())
    if missing_classes:
        raise ValueError(
            'Upload must contain all rating groups. Missing: '
            + ', '.join(sorted(missing_classes))
        )

    # Stratified splitting needs at least two reviews in every class.
    if (data['Class'].value_counts() < 2).any():
        raise ValueError('Each sentiment class needs at least two reviews.')

    audit_table = pd.DataFrame(
        audit, columns=['Processing stage', 'Rows remaining']
    )
    return raw, data.reset_index(drop=True), audit_table



# 7. READ TXT, CSV OR PDF PREDICTION INPUTS

def read_prediction_file(uploaded_file):
    """Return review texts extracted from a user-uploaded prediction file."""
    if uploaded_file is None:
        return []

    file_bytes = uploaded_file.getvalue()             # Read upload once.
    filename = uploaded_file.name.lower()

    if filename.endswith('.txt'):
        text = file_bytes.decode('utf-8', errors='ignore').strip()
        return [text] if text else []

    if filename.endswith('.csv'):
        uploaded_data = pd.read_csv(BytesIO(file_bytes))
        if 'text' in uploaded_data.columns:
            column = 'text'
        else:
            text_columns = uploaded_data.select_dtypes(
                include=['object', 'string']
            ).columns
            if len(text_columns) == 0:
                raise ValueError('Prediction CSV needs a text column.')
            column = text_columns[0]
        reviews = uploaded_data[column].dropna().astype(str).str.strip()
        return reviews[reviews != ''].tolist()

    if filename.endswith('.pdf'):
        reader = PdfReader(BytesIO(file_bytes))
        reviews = []
        for page in reader.pages:                     # One review per PDF page.
            page_text = (page.extract_text() or '').strip()
            if page_text:
                reviews.append(page_text)
        return reviews

    raise ValueError('Only TXT, CSV and PDF files are accepted.')



# 8. TRAIN TF-IDF, SVM AND RANDOM FOREST

@st.cache_resource(show_spinner=False, max_entries=3)
def train_models(text_values, class_values):
    """Train both models after SMOTE, and evaluate on untouched test data."""
    X = pd.Series(text_values)
    y = pd.Series(class_values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Learn TF-IDF vocabulary from training reviews only.
    tfidf = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2), max_features=10000
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Count the real training reviews before generating synthetic examples.
    before_smote = y_train.value_counts().reindex(CLASS_NAMES, fill_value=0)
    # SMOTE needs at least two real training rows in each class. A smaller
    # neighbourhood lets compatible small user uploads work too.
    smallest_class = int(before_smote.min())
    if smallest_class < 2:
        raise ValueError(
            'SMOTE needs at least two training reviews in each sentiment class. '
            'Please upload more labelled reviews.'
        )
    neighbours = min(5, smallest_class - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=neighbours)
    # Resample the training TF-IDF matrix only. Never fit SMOTE on the test set.
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_tfidf, y_train
    )
    after_smote = pd.Series(y_train_balanced).value_counts().reindex(
        CLASS_NAMES, fill_value=0
    )
    balance_table = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Training before SMOTE': before_smote.to_numpy(),
        'Training after SMOTE': after_smote.to_numpy(),
    })

    # Build only the two algorithms assigned to Group 4.
    # SMOTE already balances the training rows, so no extra class weighting.
    svm = LinearSVC(random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    svm.fit(X_train_balanced, y_train_balanced)      # Train balanced SVM.
    forest.fit(X_train_balanced, y_train_balanced)  # Train balanced forest.

    models = {'SVM': svm, 'Random Forest': forest}
    predictions = {
        'SVM': svm.predict(X_test_tfidf),
        'Random Forest': forest.predict(X_test_tfidf),
    }
    scores = {
        'SVM': svm.decision_function(X_test_tfidf),
        'Random Forest': forest.predict_proba(X_test_tfidf),
    }
    y_binary = label_binarize(y_test, classes=CLASS_NAMES)

    rows = []
    for name in models:
        predicted = predictions[name]
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, predicted),
            'Precision': precision_score(
                y_test, predicted, average='macro', zero_division=0
            ),
            'Recall': recall_score(
                y_test, predicted, average='macro', zero_division=0
            ),
            'F1 Score': f1_score(
                y_test, predicted, average='macro', zero_division=0
            ),
            'ROC-AUC': roc_auc_score(y_binary, scores[name], average='macro'),
        })

    return {
        'tfidf': tfidf,
        'models': models,
        'predictions': predictions,
        'scores': scores,
        'metrics': pd.DataFrame(rows),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_binary': y_binary,
        'balance_table': balance_table,
        'smote_neighbours': neighbours,
    }


def get_model_results():
    """Train from an uploaded labelled CSV, or from Recipe_Dataset.csv."""

    # A labelled user upload still replaces the classroom dataset for this session.
    if st.session_state.clean_data is not None:
        data = st.session_state.clean_data
    else:
        # Read the CSV itself; there is no saved joblib model to load.
        if not TRAINING_CSV_FILE.exists():
            raise ValueError(
                'Recipe_Dataset.csv is missing. Place it beside review_sentiment.py.'
            )
        _, data, _ = process_training_csv(TRAINING_CSV_FILE.read_bytes())

    # Cached training avoids repeating this work on every Streamlit rerun.
    return train_models(tuple(data['Clean_Text']), tuple(data['Class']))


def get_corpus_data():
    """Return the active corpus, using Recipe_Dataset.csv when needed."""
    # A labelled upload has already been cleaned and stored in this session.
    if st.session_state.clean_data is not None:
        return (
            st.session_state.raw_data,
            st.session_state.clean_data,
            st.session_state.cleaning_audit,
            st.session_state.training_filename,
        )

    # Written reviews and unlabelled files still have a reference corpus.
    # Read it through the same cleaning function used for uploaded CSV files.
    if not TRAINING_CSV_FILE.exists():
        raise ValueError('Place Recipe_Dataset.csv beside review_sentiment.py.')
    raw, clean, audit = process_training_csv(TRAINING_CSV_FILE.read_bytes())
    return raw, clean, audit, TRAINING_CSV_FILE.name



# 9. CREATE SIMPLE PDF DOWNLOADS IN MEMORY

def dataframe_text(dataframe, maximum_rows=35):
    """Turn a dataframe into readable fixed-width report text."""
    return dataframe.head(maximum_rows).to_string(index=False)


def create_pdf(title, sections):
    """Create a basic multi-page PDF without saving it on the server."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for heading, content in sections:
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.text(0.08, 0.95, title, fontsize=18, fontweight='bold')
            figure.text(0.08, 0.90, heading, fontsize=13, fontweight='bold')

            wrapped_lines = []
            for line in str(content).splitlines():
                wrapped_lines.extend(textwrap.wrap(line, 95) or [''])
            visible_text = '\n'.join(wrapped_lines[:48])

            figure.text(
                0.08, 0.86, visible_text,
                fontsize=9, family='monospace', va='top'
            )
            plt.axis('off')
            pdf.savefig(figure, bbox_inches='tight')
            plt.close(figure)

    buffer.seek(0)
    return buffer.getvalue()


def pdf_button(title, sections, filename, key):
    """Display a consistent PDF download button."""
    st.download_button(
        'Download results as PDF',
        data=create_pdf(title, sections),
        file_name=filename,
        mime='application/pdf',
        icon=':material/download:',
        key=key,
    )



# PAGE 1 - LANDING PAGE COMBINED WITH PREDICT A REVIEW

def landing_page():
    """Introduce the project and accept typed/uploaded prediction content."""
    # The four highlighted dataset metric cards were removed from this page.
    st.title('Restaurant / recipe review sentiment analytics')
    st.caption('An upload-driven Advanced Text Analytics project by Group 4')
    st.markdown(
        ':blue-badge[TF-IDF] :green-badge[SVM] '
        ':orange-badge[Random Forest] :violet-badge[Three classes]'
    )

    with st.container(border=True):
        st.subheader('How to use the application')
        st.write(
            'The results pages initially describe Recipe_Dataset.csv. You can '
            'upload a labelled CSV to replace that training corpus, or type a '
            'review or upload TXT, CSV or PDF content for prediction.'
        )

    # Clearly state the required structure before the user selects a dataset.
    # This helps prevent incompatible files from being uploaded.
    st.subheader('Required training dataset format')
    st.markdown(
        '''
The uploaded training dataset must be a **CSV file** containing these columns:

- `text` - the written restaurant or recipe review.
- `stars` - the numerical rating from 0 to 5.

The application creates the labels automatically:

- `4-5` stars = **positive**
- `3` stars = **neutral**
- `1-2` stars = **negative**
- `0` stars = **excluded**
        '''
    )

    st.info(
        '**Rating guide:** 4-5 = positive - 3 = neutral - '
        '1-2 = negative - 0 = excluded'
    )

    if st.session_state.clean_data is None:
        st.info(
            'Models train from Recipe_Dataset.csv in this folder, with SMOTE '
            'applied only to the training split. The results pages show this '
            'dataset until you upload a labelled CSV.'
        )
    else:
        st.success(
            f"Training corpus ready: {st.session_state.training_filename}. "
            'You may now change models without uploading the dataset again.'
        )

    st.subheader('Predict sentiment')
    with st.form('prediction_form', border=True):
        # One intelligent uploader accepts either a labelled training dataset
        # or unlabelled reviews. CSV structure determines how it is treated.
        uploaded_file = st.file_uploader(
            'Upload training dataset or review file',
            type=['csv', 'txt', 'pdf'],
            help=(
                'CSV with text + stars trains the models. CSV with text only, '
                'TXT and PDF are treated as reviews for prediction.'
            ),
            key='landing_file_uploader',
        )

        written_text = st.text_area(
            'Write review text', height=160,
            placeholder='The recipe was simple and tasted wonderful.'
        )

        # Give users a separate prediction button for each required model.
        svm_column, forest_column, details_column = st.columns(3)
        with svm_column:
            predict_svm = st.form_submit_button(
                'Predict with SVM', type='primary',
                icon=':material/analytics:', width='stretch'
            )
        with forest_column:
            predict_forest = st.form_submit_button(
                'Predict with Random Forest', type='primary',
                icon=':material/forest:', width='stretch'
            )
        with details_column:
            explore_details = st.form_submit_button(
                'Explore details', icon=':material/bar_chart:', width='stretch'
            )

    # Explore details uses the existing predictions and never retrains models.
    if explore_details:
        if st.session_state.prediction_results is None:
            st.warning('Predict sentiment first, then explore the detailed results.')
        else:
            st.switch_page(exploratory_page)

    # Either model-specific button starts the common prediction workflow.
    predict = predict_svm or predict_forest

    if predict:
        try:
            # Clear the previous headline while a new prediction is calculated.
            st.session_state.landing_sentiment = None
            st.session_state.landing_model = None
            st.session_state.prediction_results = None
            st.session_state.prediction_input_kind = None
            # A newly uploaded file should require a fresh model selection.
            st.session_state.analytics_model_selector = None

            # Keep prediction-file reviews separate from the training dataset.
            uploaded_reviews = []

            # A written review takes priority even when a file is attached.
            # In that case, leave the attachment untouched for this prediction.
            if uploaded_file is not None and not written_text.strip():
                filename = uploaded_file.name.lower()

                # Inspect an uploaded CSV to decide whether it is labelled.
                if filename.endswith('.csv'):
                    uploaded_csv = pd.read_csv(BytesIO(uploaded_file.getvalue()))

                    # A CSV containing text and stars is the training dataset.
                    if {'text', 'stars'}.issubset(uploaded_csv.columns):
                        raw, clean, audit = process_training_csv(
                            uploaded_file.getvalue()
                        )
                        st.session_state.raw_data = raw
                        st.session_state.clean_data = clean
                        st.session_state.cleaning_audit = audit
                        st.session_state.training_filename = uploaded_file.name
                        st.session_state.prediction_results = None
                        st.session_state.prediction_source = None

                        # When there is no written input, the labelled CSV is
                        # also treated as the attachment to be summarized.
                        if not written_text.strip():
                            uploaded_reviews = clean['text'].astype(str).tolist()

                    # A CSV without stars is treated as prediction content.
                    else:
                        uploaded_reviews = read_prediction_file(uploaded_file)

                # TXT and PDF files are always prediction content.
                else:
                    uploaded_reviews = read_prediction_file(uploaded_file)

            # Without an uploaded labelled CSV, the models train directly from
            # Recipe_Dataset.csv. A later labelled upload can replace it.

            reviews = []                              # Collect prediction inputs.

            # PRIORITY RULE: written text wins when both inputs are supplied.
            # The attached labelled CSV may still train the models, but its
            # reviews are not mixed with the written prediction input.
            if written_text.strip():
                reviews.append(written_text.strip())
            else:
                reviews.extend(uploaded_reviews)

            if not reviews:
                st.warning('Write review text or attach a TXT, CSV or PDF file.')
                return

            pairs = [(text, clean_text(text)) for text in reviews]
            pairs = [(original, cleaned) for original, cleaned in pairs if cleaned]
            if not pairs:
                st.warning('No usable text was found.')
                return

            results = get_model_results()             # Use uploaded training data.
            matrix = results['tfidf'].transform([item[1] for item in pairs])
            # Generate both predictions now. Exploratory analytics lets the
            # user switch between them without re-uploading or retraining.
            svm_prediction = results['models']['SVM'].predict(matrix)
            forest_prediction = results['models']['Random Forest'].predict(matrix)

            st.session_state.prediction_results = pd.DataFrame({
                'Original review': [item[0] for item in pairs],
                'Cleaned review': [item[1] for item in pairs],
                'SVM prediction': svm_prediction,
                'Random Forest prediction': forest_prediction,
            })
            st.session_state.prediction_source = (
                'Written text'
                if written_text.strip()
                else uploaded_file.name
            )
            # Detail pages must stay blank for typed-text predictions.
            st.session_state.prediction_input_kind = (
                'text' if written_text.strip() else 'file'
            )

            # Use the exact model chosen through the clicked prediction button.
            selected_model = 'SVM' if predict_svm else 'Random Forest'
            selected_predictions = (
                svm_prediction if selected_model == 'SVM' else forest_prediction
            )

            # For an attachment, mode() returns its majority sentiment class.
            majority_sentiment = pd.Series(selected_predictions).mode().iloc[0]
            st.session_state.landing_sentiment = majority_sentiment
            st.session_state.landing_model = selected_model

        except Exception as error:
            st.error(f'Prediction failed: {error}')

    # Display the simple prediction result directly beneath both buttons.
    if st.session_state.landing_sentiment is not None:
        sentiment = st.session_state.landing_sentiment
        emoji = {'positive': '😊', 'neutral': '😐', 'negative': '☹️'}[sentiment]

        with st.container(border=True, horizontal_alignment='center'):
            st.markdown(f'# {emoji}')
            st.subheader(f'{sentiment.capitalize()} sentiment')
            st.caption(
                f"Result produced with {st.session_state.landing_model}. "
                'For an attachment, this is the majority predicted sentiment.'
            )



# PAGE 2 - COMBINED CORPUS VIEW AND DATA PROCESSING

def corpus_processing_page():
    """Inspect the uploaded corpus or the local training dataset."""
    # With no user input, show the default Recipe_Dataset.csv corpus.
    st.header('Corpus view and data processing')
    if st.session_state.prediction_input_kind == 'text':
        # A typed review has no stars or labelled corpus of its own.
        # Show its actual cleaning steps instead of rows from Recipe_Dataset.csv.
        review = st.session_state.prediction_results.iloc[0]
        text_table = pd.DataFrame([{
            'Original review': review['Original review'],
            'Cleaned review': review['Cleaned review'],
            'Word count': len(review['Cleaned review'].split()),
        }])
        st.caption('Processing the written review supplied on the Landing page.')
        st.subheader('Your review before and after cleaning')
        st.dataframe(text_table, hide_index=True, width='stretch')
        st.markdown('''
1. Read the written review.
2. Convert it to lowercase and remove URLs, HTML, punctuation and extra spaces.
3. Convert the cleaned words to TF-IDF features using the trained vocabulary.
4. Ask SVM and Random Forest to predict its sentiment.
        ''')
        st.info('No star rating was supplied, so this review has no known true label.')
        pdf_button(
            'Written Review Processing',
            [('Your processed review', dataframe_text(text_table))],
            'written_review_processing.pdf', 'corpus_text_pdf'
        )
        return

    st.write('This page displays the dataset used to train the models.')

    # Text-only predictions use Recipe_Dataset.csv as their training corpus.
    raw, clean, audit, corpus_name = get_corpus_data()
    st.caption(f'Training corpus: {corpus_name}')

    with st.expander('View raw uploaded corpus'):
        st.dataframe(raw, width='stretch', height=400)

    st.subheader('Cleaning summary')
    st.dataframe(audit, hide_index=True, width='stretch')

    # SMOTE affects training rows, not the cleaned corpus or test rows.
    balance = get_model_results()['balance_table']
    st.subheader('Training class balance')
    st.dataframe(balance, hide_index=True, width='stretch')
    st.caption('SMOTE creates synthetic training examples after the train/test split. '
               'The test set and corpus distributions remain original.')

    st.subheader('Processing steps')
    st.markdown('''
1. Read the active CSV (Recipe_Dataset.csv by default).
2. Validate the `text` and `stars` columns.
3. Remove missing, blank and duplicate reviews.
4. Exclude 0-star and invalid ratings.
5. Clean review text with regular expressions.
6. Create negative, neutral and positive labels.
7. Prepare cleaned text for TF-IDF modelling.
8. Apply SMOTE to training TF-IDF features only; keep test reviews untouched.
    ''')

    st.subheader('Before-and-after examples')
    st.dataframe(
        clean[['text', 'Clean_Text', 'stars', 'Class']].head(30),
        hide_index=True, width='stretch'
    )

    pdf_button(
        'Corpus and Data Processing Results',
        [
            ('Training source', corpus_name),
            ('Cleaning summary', dataframe_text(audit)),
            ('Training class balance', dataframe_text(balance)),
            ('Processed sample', dataframe_text(
                clean[['Clean_Text', 'stars', 'Class']], 25
            )),
        ],
        'corpus_and_processing_results.pdf', 'corpus_pdf'
    )



# PAGE 3 - EXPLORATORY ANALYTICS AND LATEST PREDICTIONS

def exploratory_analytics_page():
    """Show prediction results and analytics from the training corpus."""
    # Without a prediction, the default dataset still feeds this page.
    st.header('Exploratory analytics')

    # Model choice belongs on this page. Both trained models remain available,
    # so switching this control does not require another dataset upload.
    selected_model = st.segmented_control(
        'Select a model',
        ['SVM', 'Random Forest'],
        default='SVM' if st.session_state.prediction_results is None else None,
        key='analytics_model_selector',
    )

    # After a user prediction, keep details blank until a model is selected.
    if selected_model is None:
        return

    if st.session_state.prediction_input_kind == 'text':
        # All figures in this branch come from the user's one written review.
        # Never mix in distributions or test-set metrics from the training CSV.
        review = st.session_state.prediction_results.iloc[0]
        sentiment = review[f'{selected_model} prediction']
        cleaned = review['Cleaned review']
        word_counts = pd.Series(cleaned.split()).value_counts()
        st.subheader(f'{selected_model} result for your review')
        st.success(f'Predicted sentiment: {sentiment.capitalize()}')
        st.write(f"**Original review:** {review['Original review']}")
        st.write(f'**Cleaned review:** {cleaned}')
        st.metric('Words in your review', len(cleaned.split()))
        st.subheader('Words in your review')
        st.bar_chart(word_counts)
        st.caption('This chart counts only words in your written review.')
        pdf_button(
            'Written Review Exploratory Results',
            [
                ('Review and prediction',
                 f"Original: {review['Original review']}\n"
                 f'Cleaned: {cleaned}\n'
                 f'{selected_model} prediction: {sentiment}'),
                ('Word counts', word_counts.to_string()),
            ],
            'written_review_exploratory_results.pdf', 'analytics_text_pdf'
        )
        return

    # Retrieve results calculated from the same user-uploaded training dataset.
    model_results = get_model_results()
    model_metrics = model_results['metrics']
    selected_metrics = model_metrics[model_metrics['Model'] == selected_model].iloc[0]

    # Display the selected model's key results immediately below the selector.
    st.subheader(f'{selected_model} results')
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric('Accuracy', f"{selected_metrics['Accuracy']:.2%}", border=True)
    metric2.metric('Precision', f"{selected_metrics['Precision']:.2%}", border=True)
    metric3.metric('Recall', f"{selected_metrics['Recall']:.2%}", border=True)
    metric4.metric('F1 score', f"{selected_metrics['F1 Score']:.2%}", border=True)
    metric5.metric('ROC-AUC', f"{selected_metrics['ROC-AUC']:.2%}", border=True)

    # Prediction results appear first because Landing redirects to this page.
    prediction_view = None
    if st.session_state.prediction_results is not None:
        all_predictions = st.session_state.prediction_results
        prediction_column = f'{selected_model} prediction'

        # Show only the prediction belonging to the currently selected model.
        prediction_view = all_predictions[
            ['Original review', 'Cleaned review', prediction_column]
        ].rename(columns={prediction_column: 'Predicted sentiment'})
        prediction_view['Model'] = selected_model

        st.success(f"Prediction source: {st.session_state.prediction_source}")
        st.subheader(f'Latest {selected_model} predictions')
        st.dataframe(prediction_view, hide_index=True, width='stretch')
        st.bar_chart(prediction_view['Predicted sentiment'].value_counts())

    # For typed reviews, these charts describe Recipe_Dataset.csv, not the
    # single typed review. The prediction table above describes that review.
    _, data, _, corpus_name = get_corpus_data()
    st.caption(f'Distributions below describe the training corpus: {corpus_name}')
    left, right = st.columns(2)
    star_counts = data['stars'].value_counts().sort_index()
    class_counts = data['Class'].value_counts().reindex(CLASS_NAMES)

    with left:
        with st.container(border=True):
            st.subheader('Star-rating distribution')
            st.bar_chart(star_counts)

    with right:
        with st.container(border=True):
            st.subheader('Sentiment distribution')
            st.bar_chart(class_counts)

    words = data['Clean_Text'].str.split().explode()
    top_words = words.value_counts().head(20)
    st.subheader('Twenty most frequent cleaned words')
    st.bar_chart(top_words)

    length_stats = data['Word_Count'].describe().round(2).to_frame('Word count')
    st.subheader('Review-length statistics')
    st.dataframe(length_stats, width='stretch')

    sections = [
        ('Star ratings', star_counts.to_string()),
        ('Sentiment classes', class_counts.to_string()),
        ('Top words', top_words.to_string()),
        ('Review lengths', dataframe_text(length_stats.reset_index())),
    ]
    if prediction_view is not None:
        sections.insert(
            0,
            (f'Latest {selected_model} predictions', dataframe_text(prediction_view, 30))
        )

    # Include the currently selected model's metrics in the PDF download.
    sections.insert(
        0,
        (f'{selected_model} metrics', selected_metrics.to_string())
    )
    pdf_button(
        'Exploratory Analytics Results', sections,
        'exploratory_analytics_results.pdf', 'analytics_pdf'
    )



# PAGE 4 - MODEL EVALUATION

def model_evaluation_page():
    """Evaluate SVM and Random Forest using all required metrics."""
    # Before user input, evaluate the default Recipe_Dataset.csv corpus.
    st.header('Model evaluation')
    if st.session_state.prediction_input_kind == 'text':
        # With no known rating, we can compare model predictions but cannot
        # calculate Accuracy, Precision, Recall, F1, Confusion Matrix or ROC-AUC.
        review = st.session_state.prediction_results.iloc[0]
        comparison = pd.DataFrame({
            'Model': ['SVM', 'Random Forest'],
            'Predicted sentiment': [
                review['SVM prediction'], review['Random Forest prediction']
            ],
        })
        st.write(f"**Your review:** {review['Original review']}")
        st.subheader('Predictions from both models')
        st.dataframe(comparison, hide_index=True, width='stretch')
        agreement = review['SVM prediction'] == review['Random Forest prediction']
        st.info('The models agree on this review.' if agreement else
                'The models disagree on this review.')
        st.warning(
            'Accuracy, Precision, Recall, F1, Confusion Matrix and ROC-AUC '
            'need known true labels. A written review has no supplied star '
            'rating, so those evaluation results cannot be calculated for it.'
        )
        pdf_button(
            'Written Review Model Comparison',
            [
                ('Your review', review['Original review']),
                ('Model predictions', dataframe_text(comparison)),
                ('Evaluation note', 'No true label was supplied; test metrics '
                 'and confusion/ROC charts are unavailable for this review.'),
            ],
            'written_review_model_comparison.pdf', 'evaluation_text_pdf'
        )
        return

    with st.spinner('Training and evaluating both models...'):
        results = get_model_results()

    metrics = results['metrics']
    displayed = metrics.copy()
    metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    for column in metric_columns:
        displayed[column] = displayed[column].map(lambda value: f'{value:.2%}')

    st.subheader('Model comparison')
    st.dataframe(displayed, hide_index=True, width='stretch')
    _, _, _, corpus_name = get_corpus_data()
    st.caption(f'Training corpus: {corpus_name}. Precision, Recall and F1 use '
               'macro averaging. The held-out test set was never SMOTE-resampled.')
    st.subheader('Training class balance')
    st.dataframe(results['balance_table'], hide_index=True, width='stretch')
    st.bar_chart(metrics.set_index('Model')[metric_columns])

    selected = st.selectbox('Select a model to inspect', list(results['models']))
    predicted = results['predictions'][selected]
    y_test = results['y_test']

    report = pd.DataFrame(classification_report(
        y_test, predicted, labels=CLASS_NAMES,
        output_dict=True, zero_division=0
    )).T.round(4)
    matrix = confusion_matrix(y_test, predicted, labels=CLASS_NAMES)

    left, right = st.columns(2)
    with left:
        st.subheader(f'{selected} classification report')
        st.dataframe(report, width='stretch')
    with right:
        st.subheader(f'{selected} confusion matrix')
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            matrix, display_labels=['Negative', 'Neutral', 'Positive']
        ).plot(ax=axis, cmap='Blues', colorbar=False)
        st.pyplot(figure)

    st.subheader(f'{selected} ROC curves')
    roc_figure, roc_axis = plt.subplots(figsize=(7, 5))
    for index, class_name in enumerate(CLASS_NAMES):
        false_rate, true_rate, _ = roc_curve(
            results['y_binary'][:, index], results['scores'][selected][:, index]
        )
        roc_axis.plot(false_rate, true_rate, label=class_name.capitalize())
    roc_axis.plot([0, 1], [0, 1], '--', color='grey', label='Chance')
    roc_axis.set_xlabel('False positive rate')
    roc_axis.set_ylabel('True positive rate')
    roc_axis.legend()
    st.pyplot(roc_figure)

    matrix_table = pd.DataFrame(
        matrix,
        index=['Actual negative', 'Actual neutral', 'Actual positive'],
        columns=['Predicted negative', 'Predicted neutral', 'Predicted positive'],
    )
    pdf_button(
        'Model Evaluation Results',
        [
            ('Model comparison', dataframe_text(displayed)),
            ('Training class balance', dataframe_text(results['balance_table'])),
            (f'{selected} classification report', dataframe_text(report.reset_index())),
            (f'{selected} confusion matrix', matrix_table.to_string()),
        ],
        'model_evaluation_results.pdf', 'evaluation_pdf'
    )



# PAGE 5 - BEST MODEL INTERPRETATION

def best_model_page():
    """Choose and explain the best model using macro F1-score."""
    # With no user input, rank the models trained on Recipe_Dataset.csv.
    st.header('Best model interpretation')
    if st.session_state.prediction_input_kind == 'text':
        # A single unlabelled review cannot establish which model is better.
        # Show what each model said, without borrowing training-set rankings.
        review = st.session_state.prediction_results.iloc[0]
        svm_result = review['SVM prediction']
        forest_result = review['Random Forest prediction']
        comparison = pd.DataFrame({
            'Model': ['SVM', 'Random Forest'],
            'Prediction for your review': [svm_result, forest_result],
        })
        st.write(f"**Your review:** {review['Original review']}")
        st.dataframe(comparison, hide_index=True, width='stretch')
        if svm_result == forest_result:
            st.success(f'Both models predict {svm_result} sentiment.')
        else:
            st.warning('The two models give different sentiment predictions.')
        st.info(
            'A best model cannot be chosen from one review without its true '
            'sentiment. Provide labelled reviews to compare model performance.'
        )
        pdf_button(
            'Written Review Model Interpretation',
            [
                ('Your review', review['Original review']),
                ('Model predictions', dataframe_text(comparison)),
                ('Interpretation', 'No true sentiment label was supplied, so '
                 'neither model can be declared best for this review.'),
            ],
            'written_review_model_interpretation.pdf', 'best_text_pdf'
        )
        return

    metrics = get_model_results()['metrics']
    best_row = metrics.loc[metrics['F1 Score'].idxmax()]
    best_model = best_row['Model']
    ranked = metrics.sort_values('F1 Score', ascending=False)

    st.success(
        f"Best model: {best_model} - Macro F1: {best_row['F1 Score']:.2%}",
        icon=':material/emoji_events:'
    )
    st.write(
        'Macro F1 is used because it gives negative, neutral and positive '
        'reviews equal importance and balances precision with recall.'
    )
    st.dataframe(ranked, hide_index=True, width='stretch')

    interpretation = (
        f"{best_model} achieved the highest macro F1-score of "
        f"{best_row['F1 Score']:.4f} on the untouched test set. It is therefore "
        'selected as the best model for this dataset.'
    )
    pdf_button(
        'Best Model Interpretation',
        [('Interpretation', interpretation), ('Ranked results', dataframe_text(ranked))],
        'best_model_interpretation.pdf', 'best_pdf'
    )



# 10. TOP NAVIGATION

# Five pages remain after combining Predict with Landing and Corpus with Processing.
# Corpus and processing is intentionally placed last and remains unchanged inside.
landing = st.Page(landing_page, title='Landing page', icon=':material/home:')
corpus_page = st.Page(
    corpus_processing_page,
    title='Corpus and processing',
    icon=':material/description:',
)
exploratory_page = st.Page(
    exploratory_analytics_page,
    title='Exploratory analytics',
    icon=':material/bar_chart:',
)
evaluation_page = st.Page(
    model_evaluation_page,
    title='Model evaluation',
    icon=':material/query_stats:',
)
best_page = st.Page(
    best_model_page,
    title='Best model',
    icon=':material/emoji_events:',
)

navigation = st.navigation(
    [landing, exploratory_page, evaluation_page, best_page, corpus_page],
    position='top',
)

# Give only the final Corpus and processing navigation item a muted appearance.
# This small CSS rule changes presentation only; the page name and content remain.
st.html(
    '''
    <style>
    [data-testid="stTopNav"] li:last-child a,
    [data-testid="stTopNav"] li:last-child button {
        color: #8b949e !important;
        opacity: 0.68;
        background-color: transparent !important;
    }
    </style>
    '''
)

if st.sidebar.button('Log out', icon=':material/logout:', width='stretch'):
    st.session_state.logged_in = False
    st.rerun()

navigation.run()
