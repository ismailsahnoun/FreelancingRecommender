from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from app.recommender import base_content_recommendation, collaborative_filtering_recommendation, ensemble_recommendation
app = FastAPI()

# Define the input data model
class FreelancerRequest(BaseModel):
    freelancer_id: int
    num_projects: int = 5

# Load your model and any necessary preprocessing steps

# other_script.py

# Load the model
base_content_recommendation, collaborative_filtering_recommendation, ensemble_recommendation = joblib.load('recommendation_model.joblib')


# Define the prediction endpoint
@app.post("/predict")
def predict_recommendations(request: FreelancerRequest):
    freelancer_id = request.freelancer_id
    num_projects = request.num_projects

    # Obtain the recommendations using your ensemble_recommendation function
    recommendations = ensemble_recommendation(freelancer_id, num_projects)
    
    return {"recommendations": recommendations}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
