import numpy as np

# Global model weights for our features.
# You can initialize these to domain-informed values.
model_weights = {
    'citations': 0.5,
    'institution_funding': 0.3,
    'institution_size': 0.2
}
learning_rate = 0.01  # Step size for online updates

def get_feature_names():
    """Return the list of feature names in a fixed order."""
    return ['citations', 'institution_funding', 'institution_size']

def extract_features(paper):
    """
    Given a paper (dict), extract its features as a numpy array.
    Missing features are defaulted to 0.
    """
    feature_names = get_feature_names()
    features = np.array([paper.get(fn, 0) for fn in feature_names], dtype=float)
    return features

def get_weights_vector():
    """
    Get the current weight vector (as a numpy array) in the same order as features.
    """
    feature_names = get_feature_names()
    weights = np.array([model_weights.get(fn, 0) for fn in feature_names], dtype=float)
    return weights

def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def compute_score(paper):
    """
    Compute the paper's score using a weighted sum of features,
    passed through a sigmoid to squash the value between 0 and 1.
    """
    features = extract_features(paper)
    weights = get_weights_vector()
    raw_score = np.dot(weights, features)
    score = sigmoid(raw_score)
    return score

def rank_papers(paper_list):
    """
    Given a list of papers (each a dictionary of features),
    compute a score for each paper and return a ranked list
    (highest score first).
    """
    for paper in paper_list:
        paper['score'] = compute_score(paper)
    ranked_list = sorted(paper_list, key=lambda x: x['score'], reverse=True)
    return ranked_list

def update_model(paper, label):
    """
    Update the model weights using a simple online logistic regression update.
    
    Args:
        paper (dict): The paper for which feedback was provided.
        label (int): 1 if the paper is accepted (useful) or 0 if rejected.
    
    The update rule is:
        weights_new = weights_old + learning_rate * (label - prediction) * features
    where prediction = sigmoid(dot(weights, features))
    """
    features = extract_features(paper)
    weights = get_weights_vector()
    # Compute current prediction
    raw_score = np.dot(weights, features)
    prediction = sigmoid(raw_score)
    error = label - prediction
    # Gradient descent update
    new_weights = weights + learning_rate * error * features
    
    # Update the global model_weights dictionary accordingly
    for i, fn in enumerate(get_feature_names()):
        model_weights[fn] = new_weights[i]
    print("Updated model weights:", model_weights)

def reject_paper(paper):
    """
    Process a rejection feedback for a given paper.
    The paper is labeled with 0 (not useful) and the model is updated.
    """
    update_model(paper, label=0)

# For completeness, here is an accept function that could be used if needed.
def accept_paper(paper):
    """
    Process an acceptance feedback for a given paper.
    The paper is labeled with 1 (useful) and the model is updated.
    """
    update_model(paper, label=1)

# ---------------------------
# Example usage functions:
# ---------------------------

def process_and_rank_papers(paper_list):
    """
    This function acts as the main entry point.
    It takes a list of paper dictionaries, ranks them using the current model,
    and returns the ranked list.
    """
    ranked_papers = rank_papers(paper_list)
    return ranked_papers

def user_feedback_reject(paper):
    """
    When the user rejects a paper, call this function to update the model.
    """
    reject_paper(paper)

# ---------------------------
# Example: Testing the system
# ---------------------------
if __name__ == '__main__':
    # Create a dummy list of 100 papers with random feature values.
    # In practice, these would be your actual paper data.
    np.random.seed(42)
    papers = []
    for i in range(100):
        paper = {
            'id': i,
            'citations': np.random.randint(0, 500),
            'institution_funding': np.random.randint(0, 1000),
            'institution_size': np.random.randint(1, 50)
        }
        papers.append(paper)

    # Rank the 100 papers
    ranked = process_and_rank_papers(papers)
    print("Top 10 ranked papers:")
    for paper in ranked[:10]:
        print(f"Paper ID: {paper['id']}, Score: {paper['score']:.3f}")

    # Suppose the user rejects the paper with id 42.
    paper_to_reject = next(p for p in papers if p['id'] == 42)
    print("\nUser rejects paper with ID 42.")
    user_feedback_reject(paper_to_reject)

    # Re-rank the papers after updating the model.
    ranked_updated = process_and_rank_papers(papers)
    print("\nTop 10 ranked papers after update:")
    for paper in ranked_updated[:10]:
        print(f"Paper ID: {paper['id']}, Score: {paper['score']:.3f}")
