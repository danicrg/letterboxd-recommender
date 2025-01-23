from playwright.sync_api import sync_playwright
import pandas as pd
from collections import deque
import os
from constants import DATA_PATH

DOMAIN = "https://letterboxd.com"

RATINGS_FILE = os.path.join(DATA_PATH, "ratings.parquet")
MOVIES_FILE = os.path.join(DATA_PATH, "movies.parquet")
SCRAPED_USERS_FILE = os.path.join(DATA_PATH, "scraped_users.txt")


def transform_ratings(some_str):
    stars = {
        "★": 1,
        "★★": 2,
        "★★★": 3,
        "★★★★": 4,
        "★★★★★": 5,
        "½": 0.5,
        "★½": 1.5,
        "★★½": 2.5,
        "★★★½": 3.5,
        "★★★★½": 4.5
    }
    return stars.get(some_str, -1)


def scrape_user(username):
    """
    Scrape the user's Letterboxd page unless a CSV for that user
    already exists. If the CSV exists, return its contents.
    """
    user_csv = os.path.join(DATA_PATH, f"{username}.csv")
    if os.path.exists(user_csv):
        print(f"Reading data from {user_csv}")
        df = pd.read_csv(user_csv)
        df_user = df[['rating', 'liked', 'slug']]
        df_film = df[['title', 'link', 'img', 'slug']]
        return df_user, df_film

    print(f"Scraping {username}")
    movies_dict = {
        'title': [],
        'rating': [],
        'liked': [],
        'link': [],
        'img': [],
        'slug': []
    }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        url = f"{DOMAIN}/{username}/films/"
        page.goto(url, timeout=999999999)
        
        pagination = page.locator("li.paginate-page a")
        page_count = pagination.nth(-1).inner_text() if pagination.count() > 0 else 1
        
        for i in range(int(page_count)):
            if i > 0:
                page.goto(f"{DOMAIN}/{username}/films/page/{i + 1}", timeout=999999999)
            page.wait_for_selector("ul.poster-list")
            movies = page.locator("ul.poster-list li")
            for j in range(movies.count()):
                movie = movies.nth(j)
                movie_id = movie.locator("div[data-film-id]").get_attribute("data-film-id")
                title = movie.locator("img").get_attribute("alt")
                slug = movie.locator("div[data-film-slug]").get_attribute("data-film-slug")
                rating_element = movie.locator("p.poster-viewingdata").inner_text().strip()
                liked = movie.locator("span.like").count() > 0
                link = movie.locator("div[data-film-link]").get_attribute("data-film-link")
                img = movie.locator("img").get_attribute("src")
                
                movies_dict['title'].append(title)
                movies_dict['rating'].append(transform_ratings(rating_element))
                movies_dict['liked'].append(liked)
                movies_dict['link'].append(link)
                movies_dict['img'].append(img)
                movies_dict['slug'].append(slug)
        
        browser.close()

    # Create DataFrame and save to CSV for future use
    df = pd.DataFrame(movies_dict)
    df.to_csv(user_csv, index=False)

    df_user = df[['rating', 'liked', 'slug']].copy()
    df_film = df[['title', 'link', 'img', 'slug']]
    df_user.loc[:, "username"] = username
    return df_user, df_film


def get_recent_reviews_users(movie_slug):
    print(f"Getting recent reviews from {movie_slug}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        url = f"{DOMAIN}/film/{movie_slug}"
        page.goto(url, timeout=999999999)

        section_selector = "section#recent-reviews.film-reviews.section"
        try:
            page.wait_for_selector(section_selector, timeout=10000)
        except TimeoutError:
            print("No recent reviews section found.")
            browser.close()
            return user_names

        review_items = page.locator(f"{section_selector} ul.film-popular-review li")
        
        user_names = []
        for i in range(review_items.count()):
            review = review_items.nth(i)
            user_name = review.get_attribute("data-person")
            user_names.append(user_name)

        browser.close()
    return user_names


def load_existing_data():
    if os.path.exists(RATINGS_FILE):
        ratings_df = pd.read_parquet(RATINGS_FILE)[['rating', 'liked', 'slug', 'username']]
    else:
        ratings_df = pd.DataFrame(columns=['rating', 'liked', 'slug', 'username'])
    
    if os.path.exists(MOVIES_FILE):
        movies_df = pd.read_parquet(MOVIES_FILE)[['title', 'link', 'img', 'slug']]
    else:
        movies_df = pd.DataFrame(columns=['title', 'link', 'img', 'slug'])    
    return ratings_df.drop_duplicates(), movies_df.drop_duplicates()


def save_data(ratings_df, movies_df):
    ratings_df.to_parquet(RATINGS_FILE, index=False)
    movies_df.to_parquet(MOVIES_FILE, index=False)

def bfs_scrape(seed_user):
    """
    Scraping using BFS from the users that recently reviewed the recent movies of the seed user
    """
    # Load existing data
    ratings_df, movies_df = load_existing_data()
    q = deque([seed_user])

    while q:
        username = q.popleft()
        
        df_user, df_film = scrape_user(username)
        
        # Explore recent reviewers for each film the user has watched
        for film_slug in df_film.slug.tolist():
            # Keep the queue from ballooning too large
            if len(q) > 10:
                break
            recent_users = get_recent_reviews_users(film_slug)
            for user in recent_users:
                q.append(user)
        
        # Combine scraped data with our master DataFrames
        ratings_df = pd.concat([ratings_df, df_user]).drop_duplicates().reset_index(drop=True)
        movies_df = pd.concat([movies_df, df_film]).drop_duplicates().reset_index(drop=True)
        
        # Save updated data and user list
        save_data(ratings_df, movies_df)
        print(f"Number of movies: {len(movies_df)}")
        print(f"Number of ratings: {len(ratings_df)}")


