#set -x
dt=`date '+%Y%m%d%H'`

if [ $# -ne 1 ]
then
echo "missing the expected arguments, usage: sfm.sh https://s3.amazonaws.com/in.inceptez.bucket1/retailsrc/trans"
exit 0
fi

if [ -d /tmp/clouddata ]
then
echo "src dir is present"
else
mkdir /tmp/clouddata
fi

if [ -d /tmp/clouddata/archive ]
then
echo "archival path exists"
else
mkdir /tmp/clouddata/archive
fi


echo "importing data from cloud" >> /tmp/sfm_${dt}.log
wget $1 -O /tmp/clouddata/trans
#wget https://s3.amazonaws.com/in.inceptez.bucket1/retailsrc/trans -O /tmp/clouddata/trans

if [ $? -eq 0 ]
then
echo "import of data from cloud is completed" >> /tmp/sfm_${dt}.log
else
echo "no data imported from cloud"
exit 0
fi

if [ -f /tmp/clouddata/trans ]
then
 echo "file is present, proceeding further" >> /tmp/sfm_${dt}.log
 mv /tmp/clouddata/trans /tmp/clouddata/trans_${dt}
 trlcnt=`tail -1 /tmp/clouddata/trans_${dt} | awk -F'|' '{ print $2 }'`
 filecnt=`cat /tmp/clouddata/trans_${dt} | wc -l`
 echo "trailer count is $trlcnt"
 echo "file count is $filecnt"
 if [ $trlcnt -ne $filecnt ]
 then
 echo "moving to reject, file is invalid" >> /tmp/sfm_${dt}.log
 mkdir -p /tmp/clouddata/reject
 mv /tmp/clouddata/trans_${dt} /tmp/clouddata/reject/
 exit 0
 fi
 echo "Remove the trailer line in the file"
 sed -i '$d' /tmp/clouddata/trans_${dt}
 hadoop fs -mkdir -p /user/hduser/clouddata/

 #Check whether the above dir is created in Hadoop
 hadoop fs -test -d /user/hduser/clouddata

 if [ $? -eq 0 ]
 then
 echo "hadoop directory is created " >> /tmp/sfm_${dt}.log
 else
 echo " failed to create the hadoop directory /user/hduser/clouddata " >> /tmp/sfm_${dt}.log
 exit 0
 fi

 hadoop fs -D dfs.block.size=67108864 -copyFromLocal -f /tmp/clouddata/trans_${dt} /user/hduser/clouddata/
 if [ $? -eq 0 ]
 then
 hadoop fs -touchz /user/hduser/clouddata/_SUCCESS
 echo "Data copied to HDFS successfully"
 echo "Data copied to HDFS successfully" >>  /tmp/sfm_${dt}.log
 else
 echo "Failed to copy data to HDFS `date` " >> /tmp/sfm_${dt}.log
 fi
 echo "moving to linux archive after compressing" >> /tmp/sfm_${dt}.log
 gzip /tmp/clouddata/trans_${dt}
 mv /tmp/clouddata/trans_${dt}.gz /tmp/clouddata/archive/
else
 echo "No data to process"
 echo "No data to process" >>  /tmp/sfm_${dt}.log
 exit 0
fi
