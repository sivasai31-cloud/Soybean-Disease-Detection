import os

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)

from werkzeug.utils import secure_filename

from database import (
    init_db,
    save_prediction,
    get_all_predictions,
    delete_prediction,
    clear_history
)

from model_utils import predict_image
from disease_data import DISEASE_INFO

# ----------------------------------------
# Flask App
# ----------------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

# Create database automatically
init_db()


# ----------------------------------------
# Helper Function
# ----------------------------------------

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ----------------------------------------
# Home Page
# ----------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    details = None
    image = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:

            error = "Please upload an image."

        else:

            file = request.files["image"]

            if file.filename == "":

                error = "Please select an image."

            elif allowed_file(file.filename):

                filename = secure_filename(file.filename)

                filepath = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                file.save(filepath)

                prediction, confidence = predict_image(filepath)

                details = DISEASE_INFO.get(
                    prediction,
                    {
                        "description": "Information not available.",
                        "symptoms": "Information not available.",
                        "treatment": "Information not available.",
                        "prevention": "Information not available."
                    }
                )

                save_prediction(
                    prediction,
                    confidence,
                    filename
                )

                image = filename

            else:

                error = "Only PNG, JPG and JPEG images are allowed."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        details=details,
        image=image,
        error=error
    )


# ----------------------------------------
# Prediction History
# ----------------------------------------

@app.route("/history")
def history():

    records = get_all_predictions()

    return render_template(
        "history.html",
        records=records
    )


# ----------------------------------------
# Delete One Record
# ----------------------------------------

@app.route("/delete/<int:prediction_id>")
def delete(prediction_id):

    delete_prediction(prediction_id)

    return redirect(
        url_for("history")
    )


# ----------------------------------------
# Clear History
# ----------------------------------------

@app.route("/clear")
def clear():

    clear_history()

    return redirect(
        url_for("history")
    )


# ----------------------------------------
# Run Flask
# ----------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )