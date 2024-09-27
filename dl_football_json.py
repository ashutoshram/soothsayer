import requests
import json
import os

# URL or API endpoint to get team data (adjust based on your API)
BASE_URL = "https://nfl-api-data.p.rapidapi.com"
headers = {
    'x-rapidapi-key': "b60f526e81msh03e03fa67f404cfp108c77jsn2fb692192b3c",
    'x-rapidapi-host': "nfl-api-data.p.rapidapi.com"
}

# Function to handle the response and save it based on team abbreviation
def save_response_by_team(response_json):
    # Extract team abbreviation from response
    team_abbreviation = response_json.get("team", {}).get("abbreviation")
    
    # Check if team abbreviation exists
    if team_abbreviation:
        # Create directory to store the JSON files, if it doesn't exist
        directory = 'teams_json'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Create a file name using the team abbreviation
        file_name = f"{directory}/{team_abbreviation}.json"
        
        # Save the response to a JSON file
        with open(file_name, 'w') as json_file:
            json.dump(response_json, json_file, indent=4)
        print(f"Saved response to {file_name}")
    else:
        print("Team abbreviation not found in response.")

# Function to fetch data for all 32 teams
def fetch_all_teams_data():
    for team_id in range(1, 33):  # Assuming 1-32 are valid team IDs
        try:
            response = requests.get(f"{BASE_URL}/nfl-team-roster?id={team_id}", headers=headers)
            if response.status_code == 200:
                response_json = response.json()
                # Save the JSON response based on team abbreviation
                save_response_by_team(response_json)
            else:
                print(f"Failed to get data for team ID: {team_id}, Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching data for team ID: {team_id}: {e}")

# Run the function to get all team data
fetch_all_teams_data()