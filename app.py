from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def categorize_popularity_score(score):
    """Categorize the popularity score into low, medium, or high."""
    if score < 0.004682:  # 50th percentile (median)
        return 'Low'
    elif score < 0.021979:  # 75th percentile
        return 'Medium'
    else:
        return 'High'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        video_views = float(request.form['video_views'])
        total_channel_subscribers = float(request.form['total_channel_subscribers'])
        total_channel_views = float(request.form['total_channel_views'])
        duration_in_seconds = float(request.form['duration_in_seconds'])
        no_of_likes = float(request.form['no_of_likes'])
        hashtags = int(request.form['hashtags'])  # Assuming hashtags is an integer count
        no_of_comments = float(request.form['no_of_comments'])
        no_of_videos_the_channel = float(request.form['no_of_videos_the_channel'])
        no_of_playlist = float(request.form['no_of_playlist'])
        community_engagement = float(request.form['community_engagement'])
        creator_gender_female = int(request.form['creator_gender'] == 'Female')
        creator_gender_male = int(request.form['creator_gender'] == 'Male')
        creator_gender_other = int(request.form['creator_gender'] == 'Other')
        subtitle_no = int(request.form['subtitle'] == 'No')
        subtitle_yes = int(request.form['subtitle'] == 'Yes')
        max_quality_1080 = int(request.form['max_quality'] == '1080')
        max_quality_1440 = int(request.form['max_quality'] == '1440')
        max_quality_2160 = int(request.form['max_quality'] == '2160')
        max_quality_240 = int(request.form['max_quality'] == '240')
        max_quality_360 = int(request.form['max_quality'] == '360')
        max_quality_480 = int(request.form['max_quality'] == '480')
        max_quality_720 = int(request.form['max_quality'] == '720')
        premiered_no = int(request.form['premiered'] == 'No')
        premiered_yes = int(request.form['premiered'] == 'Yes')

        # Calculate engagement rate
        engagement_rate = (no_of_likes + no_of_comments) / video_views * 100 if video_views > 0 else 0

        # Create a DataFrame for normalization
        df = pd.DataFrame({
            'total_channel_subscribers': [total_channel_subscribers],
            'total_channel_views': [total_channel_views],
            'engagement_rate': [engagement_rate]
        })

        # Normalize metrics: Min-Max scaling
        normalized_subscribers = (df['total_channel_subscribers'] - df['total_channel_subscribers'].min()) / (
                df['total_channel_subscribers'].max() - df['total_channel_subscribers'].min())
        normalized_views = (df['total_channel_views'] - df['total_channel_views'].min()) / (
                df['total_channel_views'].max() - df['total_channel_views'].min())
        normalized_engagement = (df['engagement_rate'] - df['engagement_rate'].min()) / (
                df['engagement_rate'].max() - df['engagement_rate'].min())

        # Prepare input for the model
        input_data = np.array([[video_views, total_channel_subscribers, total_channel_views,
                                duration_in_seconds, no_of_likes, hashtags, no_of_comments,
                                no_of_videos_the_channel, no_of_playlist, community_engagement,
                                creator_gender_female, creator_gender_male, creator_gender_other,
                                subtitle_no, subtitle_yes, max_quality_1080, max_quality_1440,
                                max_quality_2160, max_quality_240, max_quality_360,
                                max_quality_480, max_quality_720, premiered_no, premiered_yes,
                                engagement_rate, normalized_subscribers[0], normalized_views[0],
                                normalized_engagement[0]]])

        # Make prediction
        prediction = model.predict(input_data)
        popularity_score = prediction[0]

        # Categorize the popularity score
        popularity_category = categorize_popularity_score(popularity_score)

        return render_template('form.html', popularity_score=popularity_score, popularity_category=popularity_category)

    return render_template('form.html', popularity_score=None, popularity_category=None)

if __name__ == '__main__':
    app.run(debug=True)
