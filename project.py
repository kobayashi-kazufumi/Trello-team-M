from interface import get_app

if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)