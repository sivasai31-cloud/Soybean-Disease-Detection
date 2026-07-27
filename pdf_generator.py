from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()


def generate_pdf(
    filename,
    prediction,
    confidence,
    details,
    output_file
):

    doc = SimpleDocTemplate(output_file)

    story = []

    story.append(
        Paragraph(
            "<b>Soybean Disease Detection Report</b>",
            styles["Title"]
        )
    )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            f"<b>Disease:</b> {prediction}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f} %",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "<b>Description</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            details["description"],
            styles["BodyText"]
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "<b>Symptoms</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            details["symptoms"].replace("\n", "<br/>"),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "<b>Treatment</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            details["treatment"].replace("\n", "<br/>"),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "<b>Prevention</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            details["prevention"].replace("\n", "<br/>"),
            styles["BodyText"]
        )
    )

    doc.build(story)