<?php
include("dbconnect.php");
extract($_REQUEST);

$q1=mysqli_query($connect,"select * from seed_det where user='$status'");
$r1=mysqli_fetch_array($q1);
$dd=$r1['seed'];
echo $dd;
/*if($dd=="1")
{
echo "Chickpeas";
}
else if($dd=="1")
{
echo "Bengal Gram Dhal";
}
else
{
echo "None";
}*/
?>