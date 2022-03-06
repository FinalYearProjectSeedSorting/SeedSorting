<?php
include("dbconnect.php");
extract($_REQUEST);
$rdate=date("d-m-Y");
$ch1=mktime(date('h')+5,date('i')+30,date('s'));
$rtime=date('h:i:s A',$ch1);

$mq=mysqli_query($connect,"select max(id) from seed_det");
$mr=mysqli_fetch_array($mq);
$id=$mr['max(id)']+1;

$qry=mysqli_query($connect,"insert into seed_det(id,details,rdate,rtime) values($id,'$status','$rdate','$rtime')");
if($qry)
{
echo "success";
}
else
{
echo "failed";
}

?>