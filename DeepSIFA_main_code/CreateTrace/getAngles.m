function angles = getAngles(v)
%
%
%Calculates angles between all subsequent jumps within a track
%
%[angles] = get_angles(v)
%
%Input:
%   v       -   Vector with two columns [xpos,ypos] containing the positions of
%               the spots in a track
%
%Output:
%   angles  -   Angles between the lines connecting the spots defined by
%               the input coordinates.
%

angles = zeros(size(v,1)-2,1);

for q=1:size(v,1)-2
    vector1=v(q+1,:)-v(q,:);
    vector2=v(q+2,:)-v(q+1,:);
    vector1=vector1/norm(vector1);
    vector2=vector2/norm(vector2);
    signum = sign(vector2(2)*vector1(1)-vector2(1)*vector1(2));
    if imag(signum*acos(vector1*vector2'))~=0 
        %Vectors are parallel
        angles(q)=0;
    else
        angles(q)=signum*acos(vector1*vector2');
    end
end

end