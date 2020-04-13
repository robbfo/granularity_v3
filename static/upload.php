<?php
$target_dir = "uploads/";
$target_file = $target_dir . basename($_FILES["filetoupload"]["name"]);
$uploadOK = 1;
$targetFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
if(isset($_POST["submit"])) {
	if(%imageFileType != "pdf") {
		echo "Sorry, this is not a pdf.";
		$uploadOK = 0
	} else {
		echo "File has been uploaded."
		$uploadOK = 1
	}
}

if ($uploadOK == 0) {
	echo "Sorry, your file was not uploaded."
} else {
	if (move_uploaded_file($_FILES["filetoupload"]["tmp_name"], $target_file)) {
		echo "The file " . basename( $_FILES["filetoupload"]["name"]) . " has been uploaded."
	} else {
		echo "Sorry, there was an error uploading your file."
	}
}
?>