"""
main.py
-------
FastAPI application — serves the sentiment analysis model as a REST API.

Endpoints:
    GET  /health          — service health check
    POST /predict         — single review prediction
    POST /predict/batch   — batch review prediction
"""

from fastapi            import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses  import StreamingResponse
from preprocess         import preprocess
from pathlib            import Path
from collections        import Counter
from pydantic           import BaseModel, Field
from typing             import List
import time
import pandas as pd
import io
import logging
import re

from predict import predict, predict_batch

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title       = "Sentiment Analysis API",
    description = "Analyzes product reviews and classifies sentiment as Positive, Neutral, or Negative.",
    version     = "1.0.0"
)

# CORS — allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    expose_headers    = ["X-Total-Reviews", "X-Positive-Pct", "X-Neutral-Pct", "X-Negative-Pct"]
)


# ── Request / Response Schemas ──────────────────────────────────────────────

class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length  = 3,
        max_length  = 5000,
        description = "Raw review text to analyze",
        example     = "The food was absolutely amazing, best restaurant in town!"
    )

class BatchReviewRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length  = 1,
        max_length  = 100,
        description = "List of review texts (max 100 per request)",
        example     = ["Great product!", "Terrible experience.", "It was okay."]
    )

class SentimentScore(BaseModel):
    Positive : float
    Neutral  : float
    Negative : float

class PredictionResponse(BaseModel):
    sentiment  : str
    confidence : float
    scores     : dict
    input      : str
    latency_ms : float

class BatchPredictionResponse(BaseModel):
    results    : List[dict]
    total      : int
    latency_ms : float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Used by load balancers and monitoring tools.
    """
    return {
        "status"  : "healthy",
        "model"   : "TF-IDF + Logistic Regression",
        "version" : "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment for a single review.
    
    Returns:
        - sentiment   : Positive | Neutral | Negative
        - confidence  : confidence score (0-100)
        - scores      : probability for each class
        - latency_ms  : inference time in milliseconds
    """
    start = time.time()

    try:
        result = predict(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    result['latency_ms'] = round((time.time() - start) * 1000, 2)
    logger.info(f"Predicted: {result['sentiment']} ({result['confidence']}%) in {result['latency_ms']}ms")

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch_sentiment(request: BatchReviewRequest):
    """
    Predict sentiment for multiple reviews in one request.
    Max 100 reviews per request.
    
    Returns:
        - results     : list of predictions
        - total       : number of reviews processed
        - latency_ms  : total inference time
    """
    start = time.time()

    try:
        results = predict_batch(request.texts)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    latency = round((time.time() - start) * 1000, 2)
    logger.info(f"Batch predicted {len(results)} reviews in {latency}ms")

    return {
        "results"    : results,
        "total"      : len(results),
        "latency_ms" : latency
    }

@app.post("/predict/file", tags=["Batch"])
async def predict_from_file(file: UploadFile = File(...)):
    """
    Accept a CSV file with a 'review' column.
    Returns a CSV with sentiment, confidence and scores added.
    Max recommended: 50,000 reviews per file.
    """
    # validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        # try utf-8 first, fall back to latin-1
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding='utf-8', engine='python')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding='latin-1', engine='python')
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV file: {str(e)}")

    if 'review' not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="CSV must have a column named 'review'."
        )

    if len(df) > 50000:
        raise HTTPException(
            status_code=422,
            detail="Maximum 50,000 reviews per file."
        )

    # drop empty reviews
    df = df.dropna(subset=['review']).reset_index(drop=True)

    # run predictions
    sentiments, confidences, pos_scores, neu_scores, neg_scores = [], [], [], [], []

    for text in df['review']:
        try:
            result = predict(str(text))
            sentiments.append(result['sentiment'])
            confidences.append(result['confidence'])
            pos_scores.append(result['scores'].get('Positive', 0))
            neu_scores.append(result['scores'].get('Neutral', 0))
            neg_scores.append(result['scores'].get('Negative', 0))
        except Exception:
            sentiments.append('Error')
            confidences.append(0)
            pos_scores.append(0)
            neu_scores.append(0)
            neg_scores.append(0)

    # add results to dataframe
    df['sentiment']          = sentiments
    df['confidence_%']       = confidences
    df['score_positive_%']   = pos_scores
    df['score_neutral_%']    = neu_scores
    df['score_negative_%']   = neg_scores

    # summary stats
    total    = len(df)
    pos_pct  = round(df[df['sentiment'] == 'Positive'].shape[0] / total * 100, 1)
    neu_pct  = round(df[df['sentiment'] == 'Neutral'].shape[0]  / total * 100, 1)
    neg_pct  = round(df[df['sentiment'] == 'Negative'].shape[0] / total * 100, 1)

    logger.info(f"Batch file: {total} reviews — Pos:{pos_pct}% Neu:{neu_pct}% Neg:{neg_pct}%")

    # return as downloadable CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=sentiment_results.csv",
            "X-Total-Reviews"   : str(total),
            "X-Positive-Pct"    : str(pos_pct),
            "X-Neutral-Pct"     : str(neu_pct),
            "X-Negative-Pct"    : str(neg_pct),
        }
    )

@app.get("/keywords", tags=["Analysis"])
def get_keywords(top_n: int = 20):
    """
    Returns top meaningful bigrams per sentiment class grouped by aspect.
    e.g. good service, worst food, slow service
    """
    try:
        import nltk
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        from nltk.util import ngrams

        data_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'reviews_clean.csv'
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['clean_text'])

        ASPECTS = {
            'Quality'   : ['quality', 'good', 'great', 'excellent', 'poor', 'best',
                           'worst', 'amazing', 'terrible', 'outstanding', 'average',
                           'perfect', 'horrible', 'superb', 'mediocre', 'fantastic'],
            'Service'   : ['service', 'staff', 'friendly', 'rude', 'helpful', 'slow',
                           'quick', 'attentive', 'unprofessional', 'polite', 'manager',
                           'waiter', 'employee', 'server', 'crew', 'team'],
            'Food'      : ['food', 'taste', 'delicious', 'bland', 'fresh', 'cold',
                           'hot', 'flavour', 'flavor', 'tasty', 'meal', 'dish',
                           'menu', 'portion', 'ingredient', 'cooked', 'raw'],
            'Price'     : ['price', 'expensive', 'cheap', 'worth', 'value', 'cost',
                           'overpriced', 'affordable', 'reasonable', 'pricey', 'money',
                           'paid', 'charge', 'bill', 'fee', 'budget'],
            'Ambience'  : ['ambience', 'atmosphere', 'clean', 'dirty', 'cozy', 'noise',
                           'noisy', 'quiet', 'comfortable', 'crowded', 'parking',
                           'location', 'decor', 'environment', 'vibe', 'seating'],
            'Experience': ['experience', 'visit', 'recommend', 'return', 'back',
                           'disappointed', 'satisfied', 'happy', 'upset', 'surprised',
                           'expected', 'impressed', 'enjoyed', 'regret', 'loved']
        }

        result = {}

        for sentiment in ['Positive', 'Negative', 'Neutral']:
            subset = df[df['sentiment'] == sentiment]['clean_text'].sample(
                min(50000, len(df[df['sentiment'] == sentiment])), random_state=42
            ).tolist()

            # extract all bigrams from corpus
            all_bigrams = []
            for review in subset:
                tokens = str(review).lower().split()
                all_bigrams.extend([' '.join(bg) for bg in ngrams(tokens, 2)])

            bigram_counts = Counter(all_bigrams)

            # group bigrams by aspect
            aspect_result = {}
            for aspect, seeds in ASPECTS.items():
                aspect_bigrams = []
                for bigram, count in bigram_counts.most_common(100000):
                    words = bigram.split()
                    # bigram must contain at least one aspect seed word
                    if any(seed in words for seed in seeds):
                        # both words must be meaningful (length > 2)
                        if all(len(w) > 2 for w in words):
                            aspect_bigrams.append({
                                'word' : bigram,
                                'count': count
                            })
                    if len(aspect_bigrams) >= top_n:
                        break

                if aspect_bigrams:
                    aspect_result[aspect] = aspect_bigrams[:top_n]

            result[sentiment] = aspect_result

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/aspects", tags=["Analysis"])
def extract_aspects(text: str):
    """
    Extract meaningful aspect phrases from a single review.
    e.g. good service, slow service, incredible service
    """
    from nltk.util import ngrams

    ASPECTS = {
        'Quality'   : ['quality', 'good', 'great', 'excellent', 'poor', 'best',
                       'worst', 'amazing', 'terrible', 'outstanding', 'average',
                       'perfect', 'horrible', 'superb', 'mediocre', 'fantastic'],
        'Service'   : ['service', 'staff', 'friendly', 'rude', 'helpful', 'slow',
                       'quick', 'attentive', 'unprofessional', 'polite', 'manager',
                       'waiter', 'employee', 'server', 'crew', 'team'],
        'Food'      : ['food', 'taste', 'delicious', 'bland', 'fresh', 'cold',
                       'hot', 'flavour', 'flavor', 'tasty', 'meal', 'dish',
                       'menu', 'portion', 'ingredient', 'cooked', 'raw'],
        'Price'     : ['price', 'expensive', 'cheap', 'worth', 'value', 'cost',
                       'overpriced', 'affordable', 'reasonable', 'pricey', 'money',
                       'paid', 'charge', 'bill', 'fee', 'budget'],
        'Ambience'  : ['ambience', 'atmosphere', 'clean', 'dirty', 'cozy', 'noisy',
                       'quiet', 'comfortable', 'crowded', 'parking', 'location',
                       'decor', 'environment', 'vibe'],
        'Experience': ['experience', 'visit', 'recommend', 'return', 'back',
                       'disappointed', 'satisfied', 'happy', 'upset', 'surprised',
                       'expected', 'impressed', 'enjoyed', 'regret', 'loved']
    }

    DESCRIPTORS = {
        'good', 'great', 'best', 'worst', 'poor', 'bad', 'amazing', 'terrible',
        'slow', 'quick', 'fast', 'friendly', 'rude', 'clean', 'dirty', 'fresh',
        'cold', 'hot', 'expensive', 'cheap', 'quiet', 'noisy', 'comfortable',
        'excellent', 'horrible', 'perfect', 'awful', 'wonderful', 'fantastic',
        'delicious', 'bland', 'tasty', 'overpriced', 'affordable', 'crowded',
        'attentive', 'unprofessional', 'polite', 'satisfied', 'disappointed',
        'impressed', 'enjoyed', 'recommend', 'incredible', 'flawless', 'lovely'
    }

    STOPWORDS = {'and', 'the', 'was', 'were', 'are', 'but', 'for',
                 'not', 'with', 'this', 'that', 'have', 'has', 'had',
                 'its', 'our', 'your', 'their', 'from', 'even', 'though',
                 'also', 'just', 'very', 'too', 'all', 'any', 'each'}

    # clean text
    clean_text   = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    clean_text   = re.sub(r'\s+', ' ', clean_text).strip()
    tokens       = clean_text.split()
    clean_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    token_set    = set(clean_tokens)
    all_bigrams  = [' '.join(bg) for bg in ngrams(clean_tokens, 2)]

    used_phrases = set()
    detected     = {}

    for aspect, seeds in ASPECTS.items():
        matched_phrases = []

        # single word matches
        for seed in seeds:
            if seed in token_set and seed not in used_phrases:
                matched_phrases.append(seed)
                used_phrases.add(seed)

        # bigrams — both words must be seed or descriptor
        for bigram in all_bigrams:
            words = bigram.split()
            has_seed        = any(seed in words for seed in seeds)
            both_meaningful = all(w in seeds or w in DESCRIPTORS for w in words)
            if has_seed and both_meaningful:
                if bigram not in used_phrases:
                    matched_phrases.append(bigram)
                    used_phrases.add(bigram)

        if matched_phrases:
            detected[aspect] = matched_phrases

    return {
        'aspects' : detected,
        'detected': len(detected) > 0
    }