for i in $( find log/ -iname *debug*.log -size +0 );do
if [ `grep -c 'ERROR' $i` -gt 0 ];then
 echo -e "\n$i"
 grep 'ERROR' --color=auto -A 5 -B 5 $i
fi
done


find log/ -iname "*debug*.log" -size +0 | while read -r file
do
    if grep -qsm 1 'ERROR' "$file"
    then
        echo -e "\n$file"
        grep 'ERROR' --color=auto -C 5 "$file"
    fi
done
