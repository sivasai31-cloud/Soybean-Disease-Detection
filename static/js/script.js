// =========================
// Elements
// =========================

const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");

// =========================
// Preview Image
// =========================

function previewImage(event) {

    const file = event.target.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function(e){

        preview.src = e.target.result;

        preview.style.display = "block";

    }

    reader.readAsDataURL(file);

}

// =========================
// Drag Over
// =========================

dropArea.addEventListener("dragover",function(e){

    e.preventDefault();

    dropArea.style.borderColor="#81C784";

    dropArea.style.background="rgba(255,255,255,.08)";

});

// =========================
// Drag Leave
// =========================

dropArea.addEventListener("dragleave",function(){

    dropArea.style.borderColor="#4CAF50";

    dropArea.style.background="transparent";

});

// =========================
// Drop Image
// =========================

dropArea.addEventListener("drop",function(e){

    e.preventDefault();

    const files=e.dataTransfer.files;

    if(files.length===0)
        return;

    fileInput.files=files;

    const reader=new FileReader();

    reader.onload=function(event){

        preview.src=event.target.result;

        preview.style.display="block";

    }

    reader.readAsDataURL(files[0]);

});

// =========================
// Validate Image
// =========================

fileInput.addEventListener("change",function(){

    const file=fileInput.files[0];

    if(!file)
        return;

    const allowed=[
        "image/jpeg",
        "image/png",
        "image/jpg"
    ];

    if(!allowed.includes(file.type)){

        alert("Only JPG, JPEG and PNG images are allowed.");

        fileInput.value="";

        preview.style.display="none";

    }

});

// =========================
// Smooth Fade Animation
// =========================

window.addEventListener("load",function(){

    const container=document.querySelector(".container");

    container.style.opacity="0";

    container.style.transform="translateY(25px)";

    setTimeout(function(){

        container.style.transition=".7s";

        container.style.opacity="1";

        container.style.transform="translateY(0px)";

    },200);

});