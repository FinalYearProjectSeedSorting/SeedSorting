<?php
include("dbconnect.php");
extract($_REQUEST);

if($act=="d")
{
$act="";
}
mysqli_query($connect,"update seed_det set seed='$act' where user='$bc'");


?>