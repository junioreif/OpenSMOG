# Copyright (c) 2020,2021,2022,2024 The Center for Theoretical Biological Physics (CTBP) - Rice University and Northeastern University
# This file is from the OpenSMOG project, released under the MIT License. 

R"""  
The :class:`~.OpenSMOG_Reporter` class stores the potential energy information for each force applied in the system.
"""

try:
    from openmm.app import *
    from openmm import *
    import openmm.unit as unit
except:
    SBM.opensmog_quit('Failed to load OpenMM. Note: OpenSMOG requires OpenMM version 8.1.0, or newer. Check your configuration.')


import time
from sys import stdout

class forcesReporter(StateDataReporter):
    def __init__(self, file, reportInterval, forces=None, **kwargs):
        super(forcesReporter, self).__init__(file, reportInterval, **kwargs)
        self._forces = forces
    
    def _constructHeaders(self):
        headers = super()._constructHeaders()

        for n in self._forces:
            headers.append(str(n))

        return headers
    def _constructReportValues(self, simulation, state):

        values = super()._constructReportValues(simulation, state)

        for i,n in enumerate(self._forces):
            values.append(simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

        return values

class stateReporter(StateDataReporter):
    def _constructHeaders(self):
        """Construct the headers for the CSV output

        Returns: a list of strings giving the title of each observable being reported on.
        """
        headers = []
        if self._progress:
            headers.append('Progress (%)')
        if self._step:
            headers.append('Step')
        if self._time:
            headers.append('Time')
        if self._potentialEnergy:
            headers.append('Potential Energy')
        if self._kineticEnergy:
            headers.append('Kinetic Energy')
        if self._totalEnergy:
            headers.append('Total Energy')
        if self._temperature:
            headers.append('Temperature')
        if self._volume:
            headers.append('Box Volume (nm^3)')
        if self._density:
            headers.append('Density (g/mL)')
        if self._speed:
            headers.append('Speed (x1000 R.U./day)')
        if self._elapsedTime:
            headers.append('Elapsed Time (s)')
        if self._remainingTime:
            headers.append('Time Remaining')
        return headers

    def _constructReportValues(self, simulation, state):
        """Query the simulation for the current state of our observables of interest.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation

        Returns
        -------
        A list of values summarizing the current state of
        the simulation, to be printed or saved. Each element in the list
        corresponds to one of the columns in the resulting CSV file.
        """
        values = []
        box = state.getPeriodicBoxVectors()
        volume = box[0][0]*box[1][1]*box[2][2]
        clockTime = time.time()
        if self._progress:
            values.append('%.1f%%' % (100.0*simulation.currentStep/self._totalSteps))
        if self._step:
            values.append(simulation.currentStep)
        if self._time:
            values.append(state.getTime().value_in_unit(unit.picosecond))
        if self._potentialEnergy:
            values.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._kineticEnergy:
            values.append(state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._totalEnergy:
            values.append((state.getKineticEnergy()+state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole))
        if self._temperature:
            values.append((2*state.getKineticEnergy()/(self._dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin) * 0.00831446261815 )
        if self._volume:
            values.append(volume.value_in_unit(unit.nanometer**3))
        if self._density:
            values.append((self._totalMass/volume).value_in_unit(unit.gram/unit.item/unit.milliliter))
        if self._speed:
            elapsedDays = (clockTime-self._initialClockTime)/86400.0
            elapsedNs = (state.getTime()-self._initialSimulationTime).value_in_unit(unit.nanosecond)
            if elapsedDays > 0.0:
                values.append('%.3g' % (elapsedNs/elapsedDays))
            else:
                values.append('--')
        if self._elapsedTime:
            values.append(time.time() - self._initialClockTime)
        if self._remainingTime:
            elapsedSeconds = clockTime-self._initialClockTime
            elapsedSteps = simulation.currentStep-self._initialSteps
            if elapsedSteps == 0:
                value = '--'
            else:
                estimatedTotalSeconds = (self._totalSteps-self._initialSteps)*elapsedSeconds/elapsedSteps
                remainingSeconds = int(estimatedTotalSeconds-elapsedSeconds)
                remainingDays = remainingSeconds//86400
                remainingSeconds -= remainingDays*86400
                remainingHours = remainingSeconds//3600
                remainingSeconds -= remainingHours*3600
                remainingMinutes = remainingSeconds//60
                remainingSeconds -= remainingMinutes*60
                if remainingDays > 0:
                    value = "%d:%d:%02d:%02d" % (remainingDays, remainingHours, remainingMinutes, remainingSeconds)
                elif remainingHours > 0:
                    value = "%d:%02d:%02d" % (remainingHours, remainingMinutes, remainingSeconds)
                elif remainingMinutes > 0:
                    value = "%d:%02d" % (remainingMinutes, remainingSeconds)
                else:
                    value = "0:%02d" % remainingSeconds
            values.append(value)
        return values

class SMOGMinimizationReporter(MinimizationReporter):

    # From the OpenMM cookbook: you must override the report method and it must have this signature.
    def report(self, iteration, x, grad, args):
        '''
        The report method is called every iteration of minimization.

        This organization of the args is required by OpenMM standards.

        Args:

            iteration (int): The index of the current iteration. This refers
                             to the current call to the L-BFGS optimizer.
                             Each time the minimizer increases the restraint strength,
                             the iteration index is reset to 0.

            x (array-like): The current particle positions in flattened order:
                            the three coordinates of the first particle,
                            then the three coordinates of the second particle, etc.

            grad (array-like): The current gradient of the objective function
                               (potential energy plus restraint energy) with
                               respect to the particle coordinates, in flattened order.

            args (dict): Additional statistics described above about the current state of minimization.
                         In particular:
                         “system energy”: the current potential energy of the system
                         “restraint energy”: the energy of the harmonic restraints
                         “restraint strength”: the force constant of the restraints (in kJ/mol/nm^2)
                         “max constraint error”: the maximum relative error in the length of any constraint

        Returns:
            bool : Specify if minimization should be stopped.
        '''

        # Within the report method you write the code you want to be executed at
        # each iteration of the minimization.
        # In this example we get the current energy, print it to the screen, and save it to an array.


        if iteration % self.reportInterval == 0: # only print to screen as frequently as indicated
            current_energy = args['system energy']
            print("{} {}".format(iteration,current_energy))
            atomx=[]
            if self.mintraj != None:
                atompos=[]
                for xval in x:
                    atompos.append(xval)
                    if len(atompos) == 3:
                        atomx.append(atompos)
                        atompos=[]
                self.mintraj.writeModel(atomx)
        # The report method must return a bool specifying if minimization should be stopped.
        # You can use this functionality for early termination.
        return False

