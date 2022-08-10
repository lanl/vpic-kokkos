#!/usr/bin/perl
use strict;
use warnings;

my $threads;
my $ranks; 
my $advance;
my $boundary;
my $sort;
my $cpu=0;
my $time=0.0;

while(<>)
{
    # capture perfromance information
    if($_ =~ /^\s*advance\_p\s\|.*\|\s*\d*\%\s*(\d*\.\d*\D\D\d*)\s*.*$/) 
    {
	$advance=$1
    }
    if($_ =~ /^\s*boundary\_p\s\|.*\|\s*\d*\%\s*(\d*\.\d*\D\D\d*)\s*.*$/) 
    {
	$boundary=$1
    }
    if($_ =~ /^\s*sort\_particles\s\|.*\|\s*\d*\%\s*(\d*\.\d*\D\D\d*)\s*.*$/) 
    {
	$sort=$1
    }   

    # output execution statistics for this run:  
    # $threads, #cpu, cumulative execution time 
    # fraction of total time for "advance_p", 
    # "boundary_p" and "sort_particles" operations. 
    if($time > 0.0 && $cpu > 0) 
    {
	print ("$threads \t $ranks \t $cpu \t $time \t $advance \t $boundary \t $sort \n");

	$cpu=0;

	$time=0.0;
    }
    
    # output the number of threads used during this run
    if($_ =~ /^.*Setting.*tpp.*([0-9])\s*$/)
    {
	$threads=$1;
    } 

    # output the number of ranks used during this run
    if($_ =~ /^.*Processors\:\s*([0-9]*)\s*$/)
    {
	$ranks=$1;
    }
    
    if($_ =~ /^.*Done\s*\(([0-9]*\.[0-9]*).*elapsed\).*$/)
    {
	$time=$1; 
	
	$cpu = $threads * $ranks
    }
}
