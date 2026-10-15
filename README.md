# RoadInsights dashboard

Standalone Streamlit dashboard for the CN7030 UK road accident severity project.

Before deployment, copy `interactive_dashboard_data.csv` and `roadinsights_logo.png` from the Google Drive `Assignment 4 UK Accidents` folder into this folder, alongside `app.py`.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.

Deploy on Streamlit Community Cloud: create a GitHub repository containing these files; select `app.py` as the entry point. The temporary Cloudflare tunnel is not needed.

**Note:** GitHub browser uploads have a 25 MiB per-file limit, and GitHub blocks files larger than 100 MiB. If the CSV exceeds these limits, use Git LFS or external object storage and update data loading accordingly. Do not publish sensitive data.
