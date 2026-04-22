# Webots World Files

These `.wbt` files define the webots simulation. All the files will have the same content, defining:

- Three Tesla Car nodes connecting to the `car_controller.py`.

- A Supervisor node connecting to the `supervisor_controller.py`.

- Traffic dilemmas: roads, intersection, lights, signs, etc.

### So if you edit one world file

Make sure to edit the other world files too.

### Why do all the files have the same content ?

Because we want to simulate self-play of all agents of the same algorithm within a simulation. When a specific algorithm world file is run, for example `traffic_ck.wbt`, then Car Controller will detect this world file name and create the `CKAgent` instance. Same goes for other algorithm world files.