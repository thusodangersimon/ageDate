! Name:		SSPTemplateFit.f90
! Author:	B Mongwane
! Date:		July {2009..2010}
!
!      vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
!      Copyright (C) 2010  Bishop Mongwane
!
!      This program is free software: you can redistribute it and/or modify
!      it under the terms of the GNU General Public License as published by
!      the Free Software Foundation, either version 3 of the License, or
!      (at your option) any later version.
!
!      This program is distributed in the hope that it will be useful,
!      but WITHOUT ANY WARRANTY; without even the implied warranty of
!      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!      GNU General Public License for more details.
!
!      You should have received a copy of the GNU General Public License
!      along with this program.  If not, see <http://www.gnu.org/licenses/>.
!      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE EVALUATE_RANGE(DataWavelength, WavelengthRange, Alpha, Omega)
!A subroutine to evaluate the wavelength array indices representing 
!the wavelength range selected by the user. Identifiers used are
!
!  Init               : The index representing the lower array bound
!  Finit              : The index representing the upper array bound
!  DataWavelength     : Array storing the wavelengths
!  WavelengthRange    : Two dimensional array which stores the the lower
!                       and upper bound of the wavelength range of interest
!  Alpha              : The lower bound of the wavelength range of interest
!  Omega              : The upper bound of the wavelength range of interest
!  Indices            : Two dimensional array which acts as a place  holder,
!                       it is needed by BINARY_SEARCH_REAL
!----------------------------------------------------------------------------------
    REAL(HIGH), DIMENSION(Init:Finit), INTENT(IN) :: DataWavelength
    REAL, DIMENSION(2), INTENT(IN) :: WavelengthRange
    INTEGER, INTENT(OUT) :: Alpha, Omega

    INTEGER, DIMENSION(2) :: Indices


    CALL BINARY_SEARCH_REAL(WavelengthRange(1), Indices, DataWavelength, Alpha)
    IF (Alpha == - 1) Alpha = Indices(1) 
        
    CALL BINARY_SEARCH_REAL(WavelengthRange(2), Indices, DataWavelength, Omega)
    IF (Omega == - 1) Omega = Indices(1)

    Alpha = Alpha + 6
    Omega = Omega + 6


  END SUBROUTINE EVALUATE_RANGE
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  REAL(HIGH) FUNCTION NORMALISE_SPEC(ModelFlux, DataFlux, FluxErr)
!A function which normalises the flux values of the model spectrum to match 
!those of the input spectrum. Identifiers used are:
! 
!   Alpha          : The lower bound of the wavelength range of interest
!   Omega          : The upper bound of the wavelength range of interest
!   ModelFlux      : An array storing the flux values of the model spectrum
!   DataFlux       : An array storing the flux values of the input spectrum
!   FluxErr        : An array storing the errors in the flux of the input spectrum
!   Sum_FSquared   : See Chapter 3 of thesis
!   Sum_fF         : See Chapter 3 of thesis
!
   REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(IN) :: ModelFlux, DataFlux, FluxErr
! 
..
! Local data
    REAL(HIGH) :: Sum_FSquared, Sum_fF


    Sum_FSquared = 0.0; Sum_fF = 0.0

    Sum_FSquared = SUM(ModelFlux(Alpha:Omega) ** 2  / FluxErr(Alpha:Omega) ** 2)
    Sum_fF = SUM((DataFlux(Alpha:Omega) * ModelFlux(Alpha:Omega)) / FluxErr(Alpha:Omega) ** 2)

    NORMALISE_SPEC = Sum_fF / Sum_FSquared
    
  END FUNCTION NORMALISE_SPEC

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  FUNCTION READ_LIBRARY(M, N)
!A function to read in the names of the spectrum templates (representing the SSP)
!models. Identifiers used are:
!   M     : Integer constant representing the extent in length, 
!           of one side of the Grid
!   N     : Integer constant representing the extent in length, 
!           of one side of the Grid
!   I,J   : Indices
!-----------------------------------------------------------------------------
    CHARACTER(24), DIMENSION(M, N) :: READ_LIBRARY 
    INTEGER, INTENT(IN) :: M, N

    INTEGER :: I, J

    REWIND(10)

    DO J = 1, N
       DO I = 1, M
          READ(10, *, IOSTAT = InputStatus), READ_LIBRARY(I, J)
       END DO
    END DO

  END FUNCTION READ_LIBRARY

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  CHARACTER(24) FUNCTION MAKE_FILE_NAME(Met, Age)
!A function which combines the proposed metallacity and age to make the 
!corresponding proposed SSP file name. Identifiers used:
!
!   Met  : the proposed metallicity
!   Age  : the proposed age
!   
!--------------------------------------------------------------------------
    REAL, INTENT(IN) :: Met, Age

     MAKE_FILE_NAME = "ssp_" // &
          TRIM(NUM2STR(Met, "(F6.4)")) // "_" // &
          TRIM(NUM2STR(Age, "(F8.6)")) // ".spec"

  END FUNCTION MAKE_FILE_NAME

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  CHARACTER(8) FUNCTION NUM2STR(Num, FormatDescriptor)
!A function to convert a real number into a string / character. Identifiers
!used are:
! 
!   Num   : the real number to be converted
!--------------------------------------------------------------------------
    REAL, INTENT(IN) :: Num
    CHARACTER(*), INTENT(IN) :: FormatDescriptor
 

    WRITE(NUM2STR, FormatDescriptor), Num

  END FUNCTION NUM2STR

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  REAL FUNCTION STR2NUM(Str, FormatDescriptor)
!A function to convert a string / character into a real number. Identifiers
!used are: 
!
!   Str   : the string / charater to be converted.
!---------------------------------------------------------------------------
    CHARACTER(*), INTENT(IN) :: Str
    CHARACTER(*), INTENT(IN) :: FormatDescriptor


    READ(Str, FormatDescriptor), STR2NUM

  END FUNCTION STR2NUM

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  LOGICAL FUNCTION OUT_OF_BOUNDS(Param, Min, Max)
!A function to determine whether a given value id out of bounds. Identifers
!used are:
!
!   Param  : The value to be evaluated
!   Min    : The lower bound
!   Max    : The upper bound
!------------------------------------------------------------------------
    REAL, INTENT(IN) :: Param, Min, Max

    IF (Param < Min .OR. Param > Max) THEN
       OUT_OF_BOUNDS = .TRUE.
    ELSE
       OUT_OF_BOUNDS = .FALSE.
    END IF

  END FUNCTION OUT_OF_BOUNDS

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE INITIALISE(X)
!A subroutine to initialise the metallicity and age of an SSP. Identifiers
!used are: 
!
!   X   : An array to store the age and metallicity parameters
!
!--------------------------------------------------------------------------
    REAL, DIMENSION(NumPar), INTENT(INOUT) :: X
    
    CALL RANDOM_NUMBER(X)
  
    X(1) = 1.0000 + (2.6989 * X(2))
    X(2) = 4.100002 + (5.06879 * X(2))

  END SUBROUTINE INITIALISE

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE RANDOM_SEEDS()
!A subroutine to use the system clock to seed random numbers. Identifiers
!used are:
!
!   IJ, N, Clock  : Integers
!---------------------------------------------------------------------------
    INTEGER, DIMENSION(:), ALLOCATABLE :: Seed
    INTEGER :: IJ, N, Clock

    CALL RANDOM_SEED(SIZE = N)
    ALLOCATE(Seed(N))
    CALL SYSTEM_CLOCK(COUNT = Clock)

    Seed = Clock + 37 * (/(IJ - 1, IJ = 1, N)/)

    CALL RANDOM_SEED(PUT = Seed)
    IF (ALLOCATED(Seed)) DEALLOCATE(Seed)

  END SUBROUTINE RANDOM_SEEDS

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SUBROUTINE RANDOM_NORMAL(InputVector)
!A subroutine to produce normally distributed random numbers from 
!uniformly distributed numbers using the Box-Muller algorithm. Identifiers
!used are:
!
!   InputVector   : An array to store the normally distributed numbers
!   PI            : The constant pi
!   X,Y           : Place holders for the array InputVector
!   I             : Integer counter
!   Len           : The size of InputVector
!
!-------------------------------------------------------------------------   
    REAL, DIMENSION(:), INTENT(INOUT) :: InputVector

! .. local data
    REAL, PARAMETER :: PI = 3.141592654
    REAL :: X = 0.0, Y = 0.0
    INTEGER :: I, Len

    Len = SIZE(InputVector)
    CALL RANDOM_NUMBER(InputVector)
    
    DO I = 1, Len - 1, 2
       CALL RANDOM_NUMBER(X); CALL RANDOM_NUMBER(Y)
       InputVector(I) = SQRT(-2.0 * LOG(X)) * COS(2.0 * PI * Y)
       InputVector(I + 1) = SQRT(-2.0 * LOG(X)) * SIN(2.0 * PI * Y)
    END DO

!   special case if Len is odd
    IF (MOD(Len, 2) /= 0) THEN
       CALL RANDOM_NUMBER(X); CALL RANDOM_NUMBER(Y)
       InputVector(Len) = SQRT(-2.0 * LOG(X)) * COS(2.0 * PI * Y)
    END IF

  END SUBROUTINE RANDOM_NORMAL


!~~~~~~~~~~~~~~~~~~ OPEN_FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!This subroutine open files and links them to a unit number for I/O purposes. 
!Identifiers used:
!
!   UnitNumber  : A unit number to link with file name
!   FileName    : Name of the file to be opened
!   Intention   : Whether the file is for reading or writing
!
!Input : UnitNumber, FileName, Intention
!Output: None
!-------------------------------------------------------------------------------------------
  SUBROUTINE OPEN_FILE(UnitNumber, FileName, Intention)

    INTEGER, INTENT(IN) :: UnitNumber
    CHARACTER(*), INTENT(IN) :: FileName, Intention
! ..
! .. local data
    INTEGER :: OpenStatus


    IF (Intention == 'Input') THEN

       OPEN(UNIT = UnitNumber, FILE = FileName, STATUS = 'OLD', IOSTAT = OpenStatus, &
            ACCESS = 'SEQUENTIAL', ACTION = 'READ', FORM = 'FORMATTED')

       IF (OpenStatus /= 0) THEN
          PRINT *, '*** Warning: Error opening file ', TRIM(FileName),' ***'
          PRINT *, '*** IOSTAT = ', OpenStatus, ' ***'

          STOP
       END IF

    ELSE

       OPEN(UNIT = UnitNumber, FILE = FileName, STATUS = 'UNKNOWN', IOSTAT = OpenStatus, &
            ACCESS = 'SEQUENTIAL', ACTION = 'READWRITE', FORM = 'FORMATTED')

       IF (OpenStatus /= 0) THEN
          PRINT *, '*** Warning: Error opening file ', TRIM(FileName),' ***'
          PRINT *, '*** IOSTAT = ', OpenStatus, ' ***'
          
          STOP
       END IF
    END IF

  END SUBROUTINE OPEN_FILE

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SUBROUTINE SEARCH_FILE(Key, Library, MdlFlux, MdlWave)
!A subroutine to search for a given SSP file in the Library and returns the contents
!i.e the flux values. Identifiers used are:
!
!   Library           : Array storing a list of file names found in the SSP library
!   Key               : The sought file
!   MdlWave           : Array to store the wavelength values of the sought file
!   MdlFlux           : Array to store the flux values of the sought file
!   XWave, XFlux      : Place holders for MdlWave and MdlFlux
!   MetIndices        : Indices used to relay positional information on the 
!                       metallicity part of the file
!   AgeIndices        : Indices used to relay positional information on the 
!                       age part of the file
!   MetPosition       : The position of the Metallicity in the grid
!   AgePosition       : The position of the age in the grid
!   Position          : Stores the position of the file in the grid
!   Index, KJ         : Array indices
!   Tag               : A character storing information on what kind of 
!                       interpolation is necessary
!--------------------------------------------------------------------------------------
    CHARACTER(24), DIMENSION(:, :), INTENT(IN) :: Library
    CHARACTER(*), INTENT(IN) :: Key
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, MdlFlux

! .. local data    
    REAL(HIGH), DIMENSION(Init:Finit) :: XWave, XFlux!necessary 
!                                        to circumvent a `segmantation fault'
    INTEGER, DIMENSION(2) :: MetIndices, AgeIndices
    INTEGER :: MetPosition, AgePosition, Position, Index, KJ
    LOGICAL :: Found
    CHARACTER :: Comments
    CHARACTER(30) :: Tag

    Found = .FALSE.
    Position = 0


    CALL LINEAR_SEARCH(Key(5:10), Library(1, :)(5:10), MetIndices, MetPosition)
    IF (MetPosition /= -1) THEN
       Index = MetPosition
    ELSE IF (ALL(MetIndices /= -1)) THEN
       Index = MetIndices(1)
    END IF


    CALL BINARY_SEARCH(Key(12:19), Library(:, Index)(12:19), &
         AgeIndices, AgePosition) 


    IF (MetPosition /= -1 .AND. AgePosition /= -1) THEN
       Found = .TRUE.

    ELSE IF (ALL(MetIndices /= -1) .AND. ALL(AgeIndices /= -1)) THEN
       Tag = "Bilinear"

    ELSE IF (MetPosition /= -1 .AND. ALL(AgeIndices /= -1)) THEN
       Tag = "Linear, along age"
       Position = MetPosition

    ELSE IF (AgePosition /= -1 .AND. ALL(MetIndices /= -1)) THEN
       Tag = "Linear, along metallicity"
       Position = AgePosition

    ELSE
       PRINT *, "Well, this is embarassing: One"
    END IF


    IF (Found) THEN
       CALL OPEN_FILE(11, Key, 'Input')
       REWIND(11)
       
       DO KJ = 7, Finit
          
          READ(11, *, IOSTAT = InputStatus), XWave(KJ), XFlux(KJ)

          IF (InputStatus < 0) EXIT
       END DO
       CLOSE(11)

       MdlWave(Alpha:Omega) = XWave(Alpha:Omega)
       MdlFlux(Alpha:Omega) = XFlux(Alpha:Omega)
       RETURN
    END IF

    CALL PREPARE_INTERPOLATION(Key, MdlWave, MdlFlux, Tag, AgeIndices, &
         MetIndices, Position, Library)

  END SUBROUTINE SEARCH_FILE

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE BINARY_SEARCH(Key, X, Indices, Pos)
!A subroutine to do the actual searching. This routine uses a binary 
!search algorithm. Identifiers used are:
!
!   X           : Array to search from
!   Key         : The sought item
!   Indices     : Integer place holders
!   Mid         : The `middle' of the array
!   KJ          : Integer counter
!   FirstInt    : The `beggining' of the array
!   LastInt     : The `end' of the array
!------------------------------------------------------------------------
    CHARACTER(*), DIMENSION(:), INTENT(IN) :: X
    CHARACTER(*), INTENT(IN) :: Key
    INTEGER, DIMENSION(:), INTENT(OUT) :: Indices
! ..
! .. local data
    INTEGER :: Mid, KJ
    INTEGER :: FirstInt, LastInt, Pos


    FirstInt = 1
    LastInt = SIZE(X)
    Pos = -1
    Indices = (/-1, -1/)

    SEARCH: DO
       Mid = (FirstInt + LastInt) / 2

       IF (Key > X(Mid)) THEN
          FirstInt = Mid + 1

          Indices(1) = Mid
          Indices(2) = Mid + 1

       ELSE IF (Key < X(Mid)) THEN
          LastInt = Mid - 1

          Indices(1) = Mid - 1
          Indices(2) = Mid

       ELSE
          Pos = Mid
          RETURN
       END IF
      
       IF (FirstInt > LastInt) EXIT
       
    END DO SEARCH
    

  END SUBROUTINE BINARY_SEARCH
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE BINARY_SEARCH_REAL(Key, Indices, X, Pos)
!
!Same as above... this routine works for real numbers only whereas the 
!one abovee works for characters only
!
!---------------------------------------------------------------------
    REAL(HIGH), DIMENSION(1:Finit - 6), INTENT(IN) :: X
    REAL, INTENT(IN) :: Key
    INTEGER :: Pos
    INTEGER, DIMENSION(1:2), INTENT(OUT) :: Indices
! ..
! .. local data
    INTEGER :: Mid, KJ
    INTEGER :: FirstInt, LastInt


    FirstInt = 1
    LastInt = SIZE(X)
    Pos = -1
    Indices = (/-1, -1/)

    SEARCH: DO
       Mid = (FirstInt + LastInt) / 2

       IF (Key > X(Mid)) THEN
          FirstInt = Mid + 1

          Indices(1) = Mid
          Indices(2) = Mid + 1

       ELSE IF (Key < X(Mid)) THEN
          LastInt = Mid - 1

          Indices(1) = Mid - 1
          Indices(2) = Mid

       ELSE
          Pos = Mid
          RETURN
       END IF
      
       IF (FirstInt > LastInt) EXIT
       
    END DO SEARCH

  END SUBROUTINE BINARY_SEARCH_REAL

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE LINEAR_SEARCH(Sought, MetLibrary, MetIndices, Pos)
!This subroutine does a linear search in the metallicity dimension. 
!Identifierss used:
!
!   Sought       : The sought item
!   MetLibrary   : An array of the different metallicities in the 
!                  library
!   MetIndices   : Integer place holders
!   Pos          : The position wheree the sought item lies
!   Extent       : The size of Metlibrary
!   K            : Integer counter
!--------------------------------------------------------------------

    CHARACTER(*), DIMENSION(:), INTENT(IN) :: MetLibrary
    CHARACTER(*), INTENT(IN) :: Sought
    INTEGER, DIMENSION(2), INTENT(OUT) :: MetIndices
    INTEGER, INTENT(OUT) :: Pos

! .. local data
    INTEGER :: K, Extent


    Extent = SIZE(MetLibrary)
    Pos = -1
    MetIndices = (/-1, -1/)

    DO K = 1, Extent - 1

       IF (Sought == MetLibrary(K)) THEN
          Pos = K

       ELSE IF (Sought > MetLibrary(K) .AND. Sought < MetLibrary(K + 1)) THEN
          MetIndices(1) = K
          MetIndices(2) = K + 1

       END IF
    END DO
 
   K = Extent

   IF (Sought == MetLibrary(K)) THEN
      Pos = K
   ELSE IF (Sought > MetLibrary(K - 1) .AND. Sought < MetLibrary(K)) THEN
      MetIndices(1) = K - 1
      MetIndices(2) = K
   END IF

!   PRINT *, MetIndices, K

 
  END SUBROUTINE LINEAR_SEARCH
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE PREPARE_INTERPOLATION(Key, MdlWave, MdlFlux, Tag, AgeIndices,&
       MetIndices, Position, Library)
!A subroutine to select between the different cases of interpolation. See
!\S 2.5 `Searching and Interpolation' of the thesis. 
! 
!
!----------------------------------------------------------------------------
    CHARACTER(*), INTENT(IN) :: Key
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, MdlFlux
    CHARACTER(30), INTENT(IN) :: Tag
    INTEGER, DIMENSION(:), INTENT(IN) :: AgeIndices, MetIndices
    INTEGER, INTENT(IN) :: Position
    CHARACTER(*), DIMENSION(:, :), INTENT(IN) :: Library
! ..
! .. local data
    CHARACTER(24) :: TLFile, TRFile, BLFile, BRFile
    CHARACTER(10) :: Flag
    INTEGER :: IK, U, K, J
    REAL :: RMKey = 0.0, RAKey = 0.0
    REAL :: Age1 = 0.0, Age2 = 0.0
    REAL :: Met1 = 0.0, Met2 = 0.0
    REAL :: Age = 0.0, Met = 0.0
    LOGICAL :: Found


    SELECT CASE(Tag)
       CASE("Bilinear")
          TLFile = Library(AgeIndices(1), MetIndices(1))
          BLFile = Library(AgeIndices(2), MetIndices(1))
          TRFile = Library(AgeIndices(1), MetIndices(2))
          BRFile = Library(AgeIndices(2), MetIndices(2))

          CALL OPEN_FILE(50, TLFile, 'Input')
          CALL OPEN_FILE(51, TRFile, 'Input')
          CALL OPEN_FILE(52, BLFile, 'Input')
          CALL OPEN_FILE(53, BRFile, 'Input')

          Met1 = STR2NUM(TLFile(5:10), "(F6.4)")
          Met2 = STR2NUM(TRFile(5:10), "(F6.4)")
          Age1 = STR2NUM(TRFile(12:19), "(F8.6)")
          Age2 = STR2NUM(BRFile(12:19), "(F8.6)")
          RMKey = STR2NUM(Key(5:10), "(F6.4)")
          RAKey = STR2NUM(Key(12:19), "(F8.4)")

          CALL BILINEAR_INTERPOLATION(RMKey, RAKey, Age1, Age2, Met1, Met2, MdlWave, MdlFlux)

          CLOSE(50); CLOSE(51); CLOSE(52); CLOSE(53)

       CASE("Linear, along metallicity")
          
          TLFile = Library(Position, MetIndices(1))
          BLFile = Library(Position, MetIndices(2))

          CALL OPEN_FILE(52, TLFile, 'Input')
          CALL OPEN_FILE(53, BLFile, 'Input')

          Met1 = STR2NUM(TLFile(5:10), "(F6.4)")
          Met2 = STR2NUM(BLFile(5:10), "(F6.4)")
          Age = STR2NUM(TLFile(12:19), "(F8.6)")
          Met = STR2NUM(Key(5:10), "(F8.6)")
          Flag = "AlongMet"

          CALL LINEAR_INTERPOLATION(Met, Met1, Met2, Age, MdlWave, MdlFlux, Flag)

          CLOSE(52); CLOSE(53)

       CASE("Linear, along age")

          TLFile = Library(AgeIndices(1), Position)
          TRFile = Library(AgeIndices(2), Position)

          CALL OPEN_FILE(52, TLFile, 'Input')
          CALL OPEN_FILE(53, TRFile, 'Input')

          Age1 = STR2NUM(TLFile(12:19), "(F8.6)")
          Age2 = STR2NUM(TRFile(12:19), "(F8.6)")
          Met = STR2NUM(TLFile(5:10), "(F6.4)")
          Age = STR2NUM(Key(12:19), "(F6.4)")
          Flag = "AlongAge"

          CALL LINEAR_INTERPOLATION(Age, Age1, Age2, Met, MdlWave, MdlFlux, Flag)

          CLOSE(52); CLOSE(53)

       CASE DEFAULT
          PRINT *, "Well, this is embarrasing: Two"
    END SELECT


  END SUBROUTINE PREPARE_INTERPOLATION

  SUBROUTINE BILINEAR_INTERPOLATION(RMKey, RAKey, Age1, Age2, Met1, Met2, MdlWave, DesiredFlux)
    REAL, INTENT(INOUT) :: RMKey, RAKey, Age1, Age2, Met1, Met2
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, DesiredFlux

! .. local data
    INTEGER :: IL, InputStatus, InputStatus1
    REAL(HIGH), DIMENSION(Init:Finit) :: TLF = 0.0, TRF = 0.0, BLF = 0.0, BRF = 0.0
    REAL(HIGH), DIMENSION(Alpha:Omega) :: MTFlux, MBFlux
    REAL :: Wavelength1 = 0.0, Wavelength2 = 0.0, Wavelength3 = 0.0, Wavelength4 = 0.0
    REAL :: DesiredParameter = 0.0, Param1 = 0.0, Param2 = 0.0



    Met1 = LOG10((10.0 ** (Met1 - 5.0)))
    Met2 = LOG10(10.0 ** (Met2 - 5.0))
    RMKey = LOG10(10.0 ** (RMKey - 5.0))
    
    Age1 =  LOG10(10.0 ** (Age1 + 1.0))
    Age2 =  LOG10(10.0 ** (Age2 + 1.0))
    RAKey = LOG10(10.0 ** (RAKey + 1.0))



    DO IL = 7, Finit

       READ(50, *, IOSTAT = InputStatus), Wavelength1, TLF(IL)
       READ(51, *, IOSTAT = InputStatus), Wavelength2, TRF(IL)
       READ(52, *, IOSTAT = InputStatus), Wavelength3, BLF(IL)
       READ(53, *, IOSTAT = InputStatus), Wavelength4, BRF(IL)


       IF (TRF(IL) == 0.0) TRF(IL) = EPSILON(TRF(IL))
       IF (TLF(IL) == 0.0) TLF(IL) = EPSILON(TLF(IL))
       IF (BRF(IL) == 0.0) BRF(IL) = EPSILON(BRF(IL))
       IF (BLF(IL) == 0.0) BLF(IL) = EPSILON(BLF(IL))


       IF(InputStatus < 0) EXIT

    END DO


    MTFlux(Alpha:Omega) = LOG10(TLF(Alpha:Omega)) + &
         (RMKey - Met1) * (LOG10(TRF(Alpha:Omega)) - LOG10(TLF(Alpha:Omega))) / (Met2 - Met1)
    MBFlux(Alpha:Omega) = LOG10(BLF(Alpha:Omega)) + &
         (RMKey - Met1) * (LOG10(BRF(Alpha:Omega)) - LOG10(BLF(Alpha:Omega))) / (Met2 - Met1)
    DesiredFlux(Alpha:Omega) = 10.0 ** (MTFlux(Alpha:Omega) + &
         (RAKey - Age1) * (MBFlux(Alpha:Omega) - MTFlux(Alpha:Omega)) / (Age2 - Age1))

   
  END SUBROUTINE BILINEAR_INTERPOLATION

  SUBROUTINE LINEAR_INTERPOLATION(X, X1, X2, Y, MdlWave, DesiredFlux, Flag)
    REAL, INTENT(INOUT) :: X, X1, X2, Y
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, DesiredFlux
    CHARACTER(*), INTENT(IN) :: Flag

! .. local data
    INTEGER :: IL, InputStatus, InputStatus1
    REAL(HIGH), DIMENSION(Init:Finit) :: FluxOne, FluxTwo
    REAL ::  Wavelength3 = 0.0


    SELECT CASE(Flag)

    CASE("AlongMet")
       X = LOG10(10.0 ** (X - 5.0))
       X1 = LOG10(10.0 ** (X1 - 5.0))
       X2 = LOG10(10.0 ** (X2 - 5.0))
       Y =  LOG10(10.0 ** (Y + 1.0))

    CASE("AlongAge")
       X =  LOG10(10.0 ** (X + 1.0))
       X1 =  LOG10(10.0 ** (X1 + 1.0))
       X2 =  LOG10(10.0 ** (X2 + 1.0))
       Y =  LOG10(10.0 ** (Y - 5.0))

    CASE DEFAULT

       WRITE(*, *), "Well, This is embarrassing: Three"
    END SELECT

    DO IL = 7, Finit
       
       READ(52, *, IOSTAT = InputStatus), Wavelength3, FluxOne(IL)
       READ(53, *, IOSTAT = InputStatus), Wavelength3, FluxTwo(IL)


       IF (FluxOne(IL) == 0.0) FluxOne(IL) = EPSILON(FluxOne(IL))
       IF (FluxTwo(IL) == 0.0) FluxTwo(IL) = EPSILON(FluxTwo(IL))

       IF(InputStatus < 0) EXIT

    END DO


    DesiredFlux(Alpha:Omega) = 10.0 ** (LOG10(FluxOne(Alpha:Omega)) + &
         (X - X1) * (LOG10(FluxTwo(Alpha:Omega)) - LOG10(FluxOne(Alpha:Omega))) / (X2 - X1))


  END SUBROUTINE LINEAR_INTERPOLATION

  SUBROUTINE MAKE_MSP(Start, Finish, NN, MSPFlux, SSPLib, ProposedParam)

    INTEGER, INTENT(IN) :: NN
    REAL, INTENT(IN) :: Start, Finish
    REAL(HIGH), DIMENSION(7:6906), INTENT(OUT) :: MSPFlux
    CHARACTER(24), DIMENSION(:), INTENT(IN) :: SSPLib
    REAL, DIMENSION(:), INTENT(INOUT) :: ProposedParam

! .. local data
    REAL :: FinishM = 0.0, MaxWidth = 0.0
    REAL, DIMENSION(NN + 1) :: Edges
    REAL, DIMENSION(NN) :: Age, AgeWidth
    


    CALL PARTITION_AGE(Start, Finish, NN, Age, AgeWidth, Edges)
    CALL GET_WEIGHT(Age, MSPFlux, SSPLib, ProposedParam, NN, AgeWidth, Edges)

  END SUBROUTINE MAKE_MSP

  SUBROUTINE PARTITION_AGE(Start, Finish, NN, Age, AgeWidth, Edges)
    REAL, INTENT(IN) :: Start, Finish
    INTEGER, INTENT(IN) :: NN
    REAL, DIMENSION(:), INTENT(OUT) :: Edges
    REAL, DIMENSION(:), INTENT(OUT) :: Age, AgeWidth

! local data
    REAL :: Step = 0.0, StartM = 0.0, FinishM = 0.0
    INTEGER :: I


! //    Step = ABS(LOG10(Finish) - LOG10(Start)) / REAL(NN)
     Step = ABS(Finish - Start) / REAL(NN)

    Edges(1) = Start

    DO I = 2, NN + 1
! //       Edges(I) = 10.0 ** (LOG10(Start) + (I - 1) * Step)
       Edges(I) = Edges(I - 1) + Step

    END DO

    DO I = 1, NN
       Age(I) = (Edges(I) + Edges(I + 1)) / 2.0
       AgeWidth(I) = Edges(I + 1) - Edges(I)

    END DO

  END SUBROUTINE PARTITION_AGE


  SUBROUTINE GET_WEIGHT(Age, MSPFlux, SSPLib, ProposedParam, NN, AgeWidth, Edges)

    REAL, DIMENSION(:), INTENT(INOUT) :: Age, AgeWidth, Edges
    REAL(HIGH), DIMENSION(7:6906), INTENT(OUT) :: MSPFlux
    CHARACTER(24), DIMENSION(:), INTENT(IN) :: SSPLib
    REAL, DIMENSION(:), INTENT(INOUT) :: ProposedParam
    INTEGER, INTENT(IN) :: NN

! .. local data
    REAL, DIMENSION(NN) :: LookBackTime, Met, W
    REAL :: Amplitude = 0.0, TStart = 0.0
    REAL :: FinalMetallicity2 = 0.0
    REAL :: AgeBegin = 0.0, AgeEnd = 0.0, FM = 0.0
    REAL, DIMENSION(NN) :: AgeTemp
! ..
    INTEGER :: I, IU, M, K



    LookBackTime(1: NN) = Age(1: NN: -1)

    W(:) = ProposedParam(1:NN) 
    Met(:) = ProposedParam(NN + 1:2 * NN)


    CALL CONSTRUCT_SPECTRUM(Met, Age, W, MSPFlux, SSPLib, NN)

  END SUBROUTINE GET_WEIGHT


  SUBROUTINE CONSTRUCT_SPECTRUM(Met, Age, W, MSPFlux, SSPLib, NN)

    REAL, DIMENSION(:), INTENT(IN) :: Met, Age, W
    REAL(HIGH), DIMENSION(7:6906), INTENT(OUT) :: MSPFlux
    CHARACTER(24), DIMENSION(:), INTENT(IN) :: SSPLib
    INTEGER, INTENT(IN) :: NN

! .. local data
    CHARACTER(6), DIMENSION(NN) :: CMet
    CHARACTER(8), DIMENSION(NN) :: CAge
    CHARACTER(24), DIMENSION(NN) :: FileNames
! ..
    REAL, DIMENSION(NN) :: ModifiedMet, ModifiedAge, LookBackTime
    REAL(HIGH), DIMENSION(7:6906, NN) :: MdlFlux, MdlWave
! ..
    INTEGER :: K, Unit = 99, J
    INTEGER :: N = 1194


    ModifiedMet = LOG10(Met * 10 ** 4) + 1
    ModifiedAge = LOG10(Age) + 8.0


    DO K = 1, SIZE(Age)

       FileNames(K) = MAKE_FILE_NAME(ModifiedMet(K), ModifiedAge(K))
       CALL BINARY_SEARCH1(FileNames(K), SSPLib, MdlFlux(:, k), MdlWave(:, k))
    END DO

 
    MSPFlux = 0.0 

    DO K = 1, SIZE(W)
       MSPFlux(:) = MSPFlux(:) + MdlFlux(:, K) * W(K) 
    END DO

  END SUBROUTINE CONSTRUCT_SPECTRUM

