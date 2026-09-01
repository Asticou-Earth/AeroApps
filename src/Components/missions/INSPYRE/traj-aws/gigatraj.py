"""
  Python cover to Gigatraj Trajectory model (written in C++).

  The main class GIGATRAJ has methods to initialize parcels for gigatraj,
running the trajectory model and plotting the results. Configuration parameters
are contained in YAML file "gigatraj.yaml". Campaign specific geometry is contained
in a separate yaml file, e.g., "inspyre.yaml".

  Arlindo da Silva, August 2026 

"""

import os
import numpy as np
import yaml
import xarray as xr

from concurrent.futures import ThreadPoolExecutor, as_completed
import shlex
import subprocess

from datetime import datetime, timedelta

class TRAJError(Exception):
    """
    Defines general exception errors.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class GIGATRAJ(object):

    def __init__(self,config='gigatraj.yaml'):
        """
        Initialize Gigatraj. 
        """

        with open("gigatraj.yaml", encoding="utf-8") as stream:
            self.cf = yaml.safe_load(stream)

        self.Verbose = self.cf['Verbose']
            
    def genParcels(self):
        """
        Generate parcels around each fire to start trajectories from.
        """

        cf, F, P, T = self.cf, self.cf['Fires'], self.cf['Parcels'], self.cf['Trajectories']

        # Create list of fires being released
        # -----------------------------------
        fires = []
        for r in cf['Trajectories']['Releases']:
                fires += r['Fires']
 
        # Create parcels for each unique fire
        # -----------------------------------
        for f in list(dict.fromkeys(fires)):

            lon, lat, dz = F[f]['lon'], F[f]['lat'], P['Vdelta_km']
            
            for z in F[f]['altitudes_km']:

                filename = f"{cf['SCRATCH']}/parcel_init.{f}.{z}km.nc" # omitting release time

                zlow, zhigh = z - dz/2, z + dz/2
            
                cmd = f"{cf['PREFIX']}/bin/gt_generate_parcels --random " + \
                      f" --clat {lat} --clon {lon} "     + \
                      f" --zlow {zlow} --zhigh {zhigh} " + \
                      f" --number {P['Number']} " + \
                      f" --radius {P['Radius_km']} --vertical {P['Vertical']} --vunits {P['Vunits']} " + \
                      f" --format {P['Format']} --netcdf {filename}"

                print(cmd+'\n')
                if os.system(cmd):
                    print("*** Error running case ***")
            
    def _prepTrajectories(self):
        """
        Prepares a list of commands to be executed later for generating Trajectories
        given parcels on files produced by genParcels().
        """

        self.TrajCmd = []
        
        cf, F, P, T = self.cf, self.cf['Fires'], self.cf['Parcels'], self.cf['Trajectories']

        self.DryRun = T['DryRun']
        
        # Loop over parcel releases
        # -------------------------
        for R in cf['Trajectories']['Releases']:

            Start, ZeroTime = R['Start'], R['Start']
            Stop = Start + timedelta(days=T['Duration_days'])
            timestep = T['Timestep_min']
            source = T['MetSource']
            MetSpec = f"ModelRun={T['ForecastCycle']}"

            dirname = f"{cf['INSPYRE']}/{T['ForecastCycle']}"
            if os.system(f"mkdir -p {dirname}"):
                raise TRAJError(f"Could not create directory {dirname}")

            # Create parcels for each unique fire
            # -----------------------------------
            for f in R['Fires']:

                for z in F[f]['altitudes_km']:

                    parcels = f"{cf['SCRATCH']}/parcel_init.{f}.{z}km.nc" # omitting release time 
                    filename = f"{dirname}/parcel_traj.{T['ForecastCycle']}.{f}.{Start.isoformat()[:-3]}_{z}km.nc"
                
                    cmd = f"{cf['PREFIX']}/bin/gtmodel_s01 --verbose" +\
                          f" --begdate {Start.isoformat()} --enddate {Stop.isoformat()}" +\
                          f" --zerodate {ZeroTime.isoformat()}" +\
                          f" --tstep {timestep}" +\
                          f" --source {source}" + \
                          f" --metoptions {MetSpec}" + \
                          f" --vertical {T['VTrace']}" +\
                          f" --parcels {parcels}" +\
                          f" --parcelvertical  {P['Vertical']}" +\
                          f" --input_netcdf " + \
                          f" --frequency {T['outFreq_hour']}" + \
                          f" --netcdf_out {filename}" \
                          f" --format \'{T['OutFormat']}\'"

                    self.TrajCmd += [cmd,]
                    
    def _listTrajFiles(self, start=None, fire=None):
        """
        Prepares a list of files for all releases.
        """

        myFiles = dict()
        myAlts  = dict()
        
        cf, F, P, T = self.cf, self.cf['Fires'], self.cf['Parcels'], self.cf['Trajectories']

        # Loop over parcel releases
        # -------------------------
        for R in cf['Trajectories']['Releases']:

            s = R['Start']

            myFiles[s] = dict()
            myAlts[s] = dict()
            dirname = f"{cf['INSPYRE']}/data/{T['ForecastCycle']}"

            # For each file
            # -------------
            for f in R['Fires']:

                myFiles[s][f] = []
                myAlts[s][f] = []
                
                for z in F[f]['altitudes_km']:

                    filename = f"{dirname}/parcel_traj.{T['ForecastCycle']}.{f}.{s.isoformat()[:-3]}_{z}km.nc"
                    
                    myFiles[s][f] += [filename,]
                    myAlts[s][f]  += [z,]

        # Optionally trim list of files
        # -----------------------------
        if start is not None:
            for s in list(myFiles.keys()):
                if s != start:
                    del myFiles[s] # drop this start
                    del myAlts[s]  # drop this start
        if fire is not None:
            for s in list(myFiles.keys()):
                 for f in list(myFiles[s].keys()):
                     if f != fire:
                         print(f"Dropping <{f}>")
                         del myFiles[s][f] # drop this fire
                         del myAlts[s][f]  # drop this fire

        return myFiles, myAlts

                    
    def genTrajectories(self):
        """
        Generate Trajectories given parcels on files produced by genParcels().
        """
        
        # Generate list of commands to be executed in parallel
        # ----------------------------------------------------
        self._prepTrajectories()

        # Dry run - Stop here
        # -------------------
        if self.DryRun:
            for cmd in self.TrajCmd:
                print(cmd+'\n')
            return

        maxConcurrent = self.cf['Trajectories']['MaxConcurrent']

        # Serial Run
        # ----------
        if maxConcurrent == 0:
            for cmd in self.TrajCmd:
                print(cmd+'\n')
                if os.system(cmd):
                    print("*** Error running case ***")
        
        # Concurrent Run
        # --------------
        else:
            with ThreadPoolExecutor(max_workers=maxConcurrent) as executor:
            
                futures = [executor.submit(run_cmd, cmd) for cmd in self.TrajCmd]

                for future in as_completed(futures):
                
                    try:
                        cmd, stdOut = future.result()
                        if self.Verbose:
                            print(cmd,'\n',stdOut)
                        
                    except subprocess.CalledProcessError as error:
                        print(f"Command failed: {error}")
                    
                    except Exception as error:
                        print(f"Unexpected error: {error}")

    def loadTrajectories(self, start=None, fire=None):
        """
        For a given Release *start* datetime, it lazy loads all netcdf files
        associated with this release date. If fire=None it loads all fires,
        otherwise, only files for the specified fire.
        Of course, the trajectories are assumed to have been calculated already
        with method genTrajectories(). 
        """

        # Get a list of files associated with this run
        # --------------------------------------------
        myFiles, myAlts = self._listTrajFiles(start,fire)
        
        # Lazy load the files, annotating with release altitude
        # -----------------------------------------------------
        Trajs = [] # flat list of Trajectories for each start/fire
        for s in myFiles:
            for f in myFiles[s]:
                traj = [] # Will hold multiple release altitudes
                for fn, z in zip(myFiles[s][f],myAlts[s][f]):
                    ds = xr.open_dataset(fn,engine='netcdf4')
                    ds.attrs['altitude_km'] = z
                    ds.attrs['fire'] = f
                    traj += [ds,]
                Trajs += [traj,]
        
        return Trajs

    def plotTrajectories(self, start=None, fire=None):
        """
        For a given Release *start* datetime, it lazy loads all netcdf files
        associated with this release date. If fire=None it loads all fires,
        otherwise, only files for the specified fire.
        Of course, the trajectories are assumed to have been calculated already
        with method genTrajectories(). 
        """
        from traj_plot import plot_traj

        cf, T = self.cf, self.cf['Trajectories']
        
        # Get a list of files associated with this run
        # --------------------------------------------
        myFiles, myAlts = self._listTrajFiles(start,fire)
        
        # Loop over and plot
        # ------------------
        dt = timedelta(hours=self.cf['Plots']['Density_timestep_hours'])
        CampaignFile = cf['Plots']['CampaignFile']
        dpi = cf['Plots']['Image_dpi']
        for s in myFiles:
            for f in myFiles[s]:

                # Bundle for this start, fire
                # ---------------------------
                Traj = [] # Will hold multiple release altitudes
                for fn, z in zip(myFiles[s][f],myAlts[s][f]):
                    ds = xr.open_dataset(fn,engine='netcdf4')
                    ds.attrs['altitude_km'] = z
                    ds.attrs['fire'] = f
                    Traj += [ds,]

                StartTime = datetime.fromisoformat(Traj[0].attrs['Trajectory_start'])
                ValidTime = StartTime + dt
                t0, tv = StartTime.isoformat()[:-3], ValidTime.isoformat()[:-3], 
 
                fig, _ = plot_traj (Traj, ValidTime, CampaignFile, satellites=None)

                dirname = f"{cf['INSPYRE']}/images/{T['ForecastCycle']}"
                img_file = f"{dirname}/parcel_traj.density.{f}.{t0}+{tv}.png"
                if cf['Verbose']:
                    print(f"[] Saving image {img_file}") 

                fig.savefig(img_file, dpi=dpi)
 
        return 
    
    
#-----
def run_cmd(cmd):
    """
    Run a single command as a subprocess.
    """
    command = shlex.split(cmd)
    result = subprocess.run(command,
                            text=True,
                            capture_output=True,
                            check=True)
    return cmd, result.stdout

#-----
if __name__ == "__main__":

    gt = GIGATRAJ()

    #myFiles = gt.loadReleases()
