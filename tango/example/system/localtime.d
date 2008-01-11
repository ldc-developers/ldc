/*******************************************************************************

        localtime.d
        
*******************************************************************************/

private import  tango.io.Stdout;

private import  tango.time.WallClock;

/******************************************************************************

        Example code to format a local time in the following format:
        "Wed Dec 31 16:00:00 GMT-0800 1969". The day and month names
        would typically be extracted from a locale instance, but we
        convert them locally here for the sake of simplicity

******************************************************************************/

void main ()
{
        /// list of day names
        static char[][] days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

        /// list of month names
        static char[][] months =
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];

        // retreive local time
        auto dt = WallClock.toDate;

        // get GMT difference in minutes
        auto tz = cast(int) WallClock.zone.minutes;
        char sign = '+';
        if (tz < 0)
            tz = -tz, sign = '-';

        // format date
        Stdout.formatln ("{}, {} {:d2} {:d2}:{:d2}:{:d2} GMT{}{:d2}:{:d2} {}",
                          days[dt.date.dow],
                          months[dt.date.month-1],
                          dt.date.day,
                          dt.time.hours, 
                          dt.time.minutes,
                          dt.time.seconds,
                          sign,
                          tz / 60,
                          tz % 60,
                          dt.date.year
                         );
}
