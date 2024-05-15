# Tailoring-Body-Measurement

This Flask backend serves as the core of a tailoring application. It enables users to measure body parts using uploaded images.

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies by running:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application by executing:
    ```
    python main.py
    ```
2. Once the server is running, you can access the endpoints via HTTP requests.

### Endpoints

- **POST `/`**: This endpoint accepts front and side images of a person's body along with additional parameters like height and gender. It processes the images and returns measurements of various body parts.

    - **Parameters**:
        - `front`: Front image of the person's body.
        - `side`: Side image of the person's body.
        - `height`: Height of the person.
        - `gender`: Gender of the person.

    - **Response**: JSON object containing measurements of different body parts.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).