from playwright.sync_api import sync_playwright
import pandas as pd

DOMAIN = "https://letterboxd.com"

def transform_ratings(some_str):
    """
    Transforms raw star rating into float value
    :param: some_str: actual star rating
    :rtype: returns the float representation of the given star(s)
    """
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
    movies_dict = {
        'id': [],
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
        
        # Navigate to user's films page
        url = f"{DOMAIN}/{username}/films/"
        page.goto(url)
        
        # Check for pagination
        pagination = page.locator("li.paginate-page a")
        page_count = pagination.nth(-1).inner_text() if pagination.count() > 0 else 1
        
        # Scrape each page
        for i in range(int(page_count)):
            if i > 0:
                page.goto(f"{DOMAIN}/{username}/films/page/{i + 1}")
            
            # Wait for the poster list to load
            page.wait_for_selector("ul.poster-list")
            
            # Extract movie details
            movies = page.locator("ul.poster-list li")
            for j in range(movies.count()):
                movie = movies.nth(j)
                movie_id = movie.locator("div[data-film-id]").get_attribute("data-film-id")  # Target specific div
                title = movie.locator("img").get_attribute("alt")
                slug = movie.locator("div[data-film-slug]").get_attribute("data-film-slug")
                rating_element = movie.locator("p.poster-viewingdata").inner_text().strip()
                liked = movie.locator("span.like").count() > 0
                link = movie.locator("div[data-film-link]").get_attribute("data-film-link")  # Specific div
                img = movie.locator("img").get_attribute("src")
                
                movies_dict['id'].append(movie_id)
                movies_dict['title'].append(title)
                movies_dict['rating'].append(transform_ratings(rating_element))
                movies_dict['liked'].append(liked)
                movies_dict['link'].append(link)
                movies_dict['img'].append(img)
                movies_dict['slug'].append(slug)
        
        browser.close()

    df_film = pd.DataFrame(movies_dict)
    
    return df_film
