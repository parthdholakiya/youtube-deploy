mkdir -p ~/.sreamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORDS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
