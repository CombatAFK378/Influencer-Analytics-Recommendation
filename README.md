# Influencer Analytics & Recommendation System

## Project Overview

This project implements a comprehensive AI/ML pipeline for influencer marketing analytics, consisting of three main components:

1. **Influencer Niche Classification** - NLP-based clustering to categorize influencers into content niches
2. **Fake Follower Detection** - Anomaly detection to identify suspicious follower growth patterns
3. **Brand-Influencer Matching** - Semantic similarity matching to recommend optimal brand partnerships

**Assessment Duration**: 3-5 hours  
**Date**: December 27, 2025

---

## Dataset Description

The project uses three CSV files containing influencer and brand data:

- **influencers_data.csv** (30 records): Influencer profiles with bio and recent captions
- **growth_data.csv** (30 records): Follower metrics, engagement data, and 30-day growth history
- **brands_data.csv** (10 records): Brand descriptions and target keywords

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

Create virtual environment
python -m venv venv

Activate virtual environment
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


---

## Project Structure

Influencer Analytics & Recommendation/
│
├── venv/ # Virtual environment
├── brands_data.csv # Input: Brand data
├── growth_data.csv # Input: Growth metrics
├── influencers_data.csv # Input: Influencer profiles
├── influencer_analytics.ipynb # Main implementation notebook
├── requirements.txt # Python dependencies
├── README.md # Documentation (this file)
│
└── outputs/ # Generated results
├── niche_classification.csv # Part 1 output
├── fake_follower_scores.csv # Part 2 output
└── brand_influencer_matches.csv # Part 3 output

---

## Methodology

### Part 1: Influencer Niche Classification

**Objective**: Automatically categorize 30 influencers into 8 meaningful content niches

**Approach**:
1. **Text Preprocessing**
   - Combined bio and recent captions into single text field
   - Cleaned text: lowercase conversion, special character removal
   - Standardized format for consistent embedding generation

2. **Semantic Embeddings**
   - Model: Sentence Transformers (`all-MiniLM-L6-v2`)
   - Generated 384-dimensional dense vectors capturing semantic meaning
   - Embeddings encode contextual relationships between words/phrases

3. **Clustering Algorithm**
   - Algorithm: K-Means with k=8 clusters
   - Initialization: k-means++ (10 iterations)
   - Distance metric: Euclidean distance in embedding space

4. **Keyword Extraction**
   - Method: TF-IDF (Term Frequency-Inverse Document Frequency)
   - Extracted top 2-3 keywords per cluster
   - Filtered stop words for meaningful descriptors

**Key Parameters**:
- Number of clusters: 8
- Embedding dimensions: 384
- Keywords per niche: 3
- Random seed: 42 (reproducibility)

**Output**: `niche_classification.csv` containing influencer IDs, assigned clusters, and niche keywords

---

### Part 2: Fake Follower Detection

**Objective**: Identify influencers with suspicious follower patterns using anomaly detection

**Approach**:
1. **Engagement Rate Calculation**
   - Formula: `(avg_likes + avg_comments) / followers × 100`
   - Benchmark: Normal range is 2-10%
   - Low engagement (<2%) indicates potential fake followers

2. **Follower Spike Detection**
   - Parsed 30-day daily follower count arrays
   - Calculated daily growth rates using first differences
   - Statistical threshold: mean + 2×standard_deviation
   - Identified days with anomalous growth exceeding threshold

3. **Multi-Factor Fake Score (0-100)**
   
   Score components:
   - **Spike Component (0-50 points)**: `min(spike_count × 10, 50)`
   - **Engagement Component (0-40 points)**:
     - <2% engagement: 40 points
     - 2-5% engagement: 30 points
     - 5-8% engagement: 15 points
     - >8% engagement: 0 points
   - **Growth Component (0-30 points)**:
     - >70% growth: 30 points
     - 50-70% growth: 20 points
     - 30-50% growth: 10 points
     - <30% growth: 0 points

4. **Reason Generation**
   - Human-readable explanations for each score
   - Flags specific red flags detected
   - Provides transparency for business decisions

**Red Flags**:
- ≥3 follower spikes in 30 days
- Engagement rate <5%
- Total growth >50% in 30 days

**Output**: `fake_follower_scores.csv` with scores, spike counts, and explanatory reasons

---

### Part 3: Brand-Influencer Matching

**Objective**: Match each of 10 brands with their top 10 most relevant influencers

**Approach**:
1. **Brand Text Preparation**
   - Combined brand description + keywords
   - Applied same preprocessing as influencer text
   - Ensured consistent embedding space

2. **Brand Embeddings**
   - Generated using same Sentence Transformer model
   - 384-dimensional vectors (aligned with influencer embeddings)
   - Captures brand identity and target audience

3. **Similarity Computation**
   - Method: Cosine similarity
   - Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
   - Range: 0 (orthogonal) to 1 (identical)
   - Converted to percentage (0-100) for interpretability

4. **Ranking & Selection**
   - Sorted influencers by similarity score (descending)
   - Selected top 10 per brand
   - Included match metadata (bio, niche, score)

**Similarity Interpretation**:
- >70: Excellent match (strong content alignment)
- 60-70: Good match (relevant audience overlap)
- 50-60: Moderate match (some alignment)
- <50: Weak match (limited relevance)

**Output**: `brand_influencer_matches.csv` with 100 total matches (10 brands × 10 influencers)

---

## How to Run

### Option 1: Run in VS Code

1. Open project folder in VS Code
2. Open `influencer_analytics.ipynb`
3. Select kernel: Choose your `venv` Python interpreter
4. Run all cells sequentially (Shift + Enter)
5. Check `outputs/` folder for results

### Option 2: Run via Jupyter Notebook

venv\Scripts\activate
Launch notebook
jupyter notebook influencer_analytics.ipynb


---

## Results Summary

### Part 1: Niche Classification
- **Influencers processed**: 30
- **Niches identified**: 8
- **Keywords per niche**: 2-3
- **Output file**: `outputs/niche_classification.csv`

### Part 2: Fake Follower Detection
- **Influencers analyzed**: 30
- **Engagement rates**: Calculated for all
- **Spikes detected**: Statistical outliers in 30-day history
- **Score range**: 0-100 (higher = more suspicious)
- **Output file**: `outputs/fake_follower_scores.csv`

### Part 3: Brand-Influencer Matching
- **Brands processed**: 10
- **Matches per brand**: 10
- **Total matches**: 100
- **Similarity metric**: Cosine similarity (0-100%)
- **Output file**: `outputs/brand_influencer_matches.csv`

---

## Technical Details

### Models & Libraries
- **NLP Model**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Clustering**: K-Means (scikit-learn)
- **Keyword Extraction**: TF-IDF (scikit-learn)
- **Similarity**: Cosine similarity (scikit-learn)

### Performance
- **Total runtime**: ~5-10 minutes (first run downloads model)
- **Embedding generation**: ~30 seconds for 40 texts
- **Clustering**: <1 second
- **Matching**: <1 second per brand

### Reproducibility
- Fixed random seed (42) for K-Means
- Deterministic model inference
- Consistent preprocessing pipeline

---

## Key Insights

1. **Niche Distribution**: Influencers cluster naturally into distinct content categories (fashion, tech, fitness, beauty, food, travel, parenting, wellness)

2. **Fake Follower Prevalence**: Statistical analysis reveals growth patterns inconsistent with organic audience building

3. **Match Quality**: Semantic embeddings successfully align brand positioning with influencer content themes

---

## Future Enhancements

- Add temporal analysis for trend detection
- Incorporate sentiment analysis on captions
- Implement multi-modal analysis (images + text)
- Build interactive dashboard for results visualization
- Add cost-effectiveness scoring (reach vs. engagement)

---

## Author

**Shashank**  
Software Developer | ML Engineer  
Mumbai, Maharashtra, India  
Date: December 27, 2025

---

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) for details.


