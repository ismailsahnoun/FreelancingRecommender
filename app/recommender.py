from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pandas as pd

talents=pd.read_csv("/code/datasets/tl.csv")
talents["skill"] = talents["skill"].apply(eval)
def to_1D(series):
 return pd.Series([x for _list in series for x in _list])
to_1D(talents["skill"])

projects = pd.read_csv('/code/datasets/pr.csv')
projects["skill"] = projects["skill"].apply(eval)
to_1D(projects["skill"])

favorites_df=pd.read_csv("/code//datasets/favorite2.csv")
favorites_df.drop(['Unnamed: 0','favorite_object_type'],axis=1, inplace=True)
favorites_df['is_favorite']=1

# Extract the unique skills from the freelancers and projects DataFrames
all_skills = set()
for skills in talents['skill']:
    all_skills.update(skills)
for skills in projects['skill']:
    all_skills.update(skills)
skills = list(all_skills)
skills.sort()
def base_content_recommendation(freelancer_id, num_projects=5):
    # Find the freelancer's skills
    freelancer_row = talents.loc[talents['talent_id'] == freelancer_id]
    freelancer_skills = freelancer_row['skill'].tolist()[0]
    
    # Create a binary vector to represent the freelancer's skills
    freelancer_vector = np.zeros(len(skills))
    for skill in freelancer_skills:
        index = skills.index(skill)
        freelancer_vector[index] = 1
    
    # Create a binary matrix to represent the required skills for each project
    project_matrix = np.zeros((len(projects), len(skills)))
    for i, row in projects.iterrows():
        project_skills = row['skill']
        for skill in project_skills:
            index = skills.index(skill)
            project_matrix[i, index] = 1
    
    # Compute the cosine similarity between the freelancer vector and each project vector
    similarities = np.dot(project_matrix, freelancer_vector) / (np.linalg.norm(project_matrix, axis=1) * np.linalg.norm(freelancer_vector))
    
    # Sort the projects by similarity and recommend the top num_projects
    recommended_project_indices = np.argsort(similarities)[::-1][:num_projects]
    recommended_projects = projects.iloc[recommended_project_indices]['desired_id'].tolist()
    return recommended_projects

# transform favorites_data to binary labels matrix
freelancer_ids = favorites_df['talent_profile_id'].unique()
project_ids = favorites_df['project_id'].unique()
binary_labels = np.zeros((len(freelancer_ids), len(project_ids)))
for i, freelancer_id in enumerate(freelancer_ids):
    freelancer_favorites = favorites_df[favorites_df['talent_profile_id'] == freelancer_id]
    liked_projects = freelancer_favorites[freelancer_favorites['is_favorite'] == 1]['project_id'].values
    binary_labels[i, np.isin(project_ids, liked_projects)] = 1
binary_labels_sparse = csr_matrix(binary_labels)


def collaborative_filtering_recommendation(freelancer_id, num_projects=5):
    # perform SVD
    svd = TruncatedSVD(n_components=10)
    svd.fit(binary_labels_sparse)

    # get latent vectors for freelancers and projects
    freelancer_latent_vectors = svd.transform(binary_labels_sparse)
    project_latent_vectors = svd.components_.T

    # compute similarity between freelancers and projects
    similarity_matrix = freelancer_latent_vectors @ project_latent_vectors.T

    # make recommendations for a freelancer
    freelancer_index = np.where(freelancer_ids == freelancer_id)[0][0]
    project_scores = similarity_matrix[freelancer_index]
    recommended_project_indices = np.argsort(project_scores)[::-1][:5]
    recommended_projects = project_ids[recommended_project_indices].tolist()
    return recommended_projects

# Ensemble recommendation using decision tree
def ensemble_recommendation(freelancer_id, num_projects=5):
    # Obtain recommendations from base-content recommendation model
    recommended_CS = base_content_recommendation(freelancer_id, num_projects)
    
    base_content_recommendations=projects.loc[projects['desired_id'].isin(recommended_CS)].client_id.tolist()
    
    # Obtain recommendations from collaborative filtering model
    collaborative_filtering_recommendations = collaborative_filtering_recommendation(freelancer_id, num_projects)

    # Ensure both models provide an equal number of recommendations
    min_num_recommendations = min(len(base_content_recommendations), len(collaborative_filtering_recommendations))

    # Truncate the recommendations to the minimum number
    base_content_recommendations = base_content_recommendations[:min_num_recommendations]
    collaborative_filtering_recommendations = collaborative_filtering_recommendations[:min_num_recommendations]

    # Create a dataset for stacking
    stacking_X = base_content_recommendations + collaborative_filtering_recommendations
    stacking_y = [1] * min_num_recommendations + [0] * min_num_recommendations


    # Train a meta-model on the entire dataset
    meta_model = DecisionTreeClassifier()
    meta_model.fit(np.array(stacking_X).reshape(-1, 1), np.array(stacking_y).reshape(-1, 1))

    ## Predict the labels for the entire dataset
    predicted_labels = meta_model.predict(np.array(stacking_X).reshape(-1, 1))

    # Combine the recommendations based on the predicted labels
    combined_recommendations = []

    for i, label in enumerate(predicted_labels):
        if label == 1:
            combined_recommendations.append(stacking_X[i])

    # Select the top num_projects projects as the final ensemble recommendations
    final_recommendations = combined_recommendations[:num_projects]
    return final_recommendations

import joblib

# ... (code for loading data and defining recommendation functions goes here)

# Train and save the ensemble recommendation model
model = ensemble_recommendation  # Replace with your model function
joblib.dump((base_content_recommendation, collaborative_filtering_recommendation, ensemble_recommendation), 'recommendation_model.joblib')
