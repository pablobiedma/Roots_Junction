package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.UserEntry;
import groupxii.database.VehicleEntry;
import groupxii.transportation.CalculatedVehicle;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;
import java.util.List;

@RestController
public class TransportationController {


    /**
     * Return a list of all available vehicle entries that the solar panel feature can process.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/vehicleNameList")
    public List<String> getNameList() {
        return Database.instance.getVehicleListVehicleNames();
    }

    /**
     * Calculate the saved CO2 and send the response to the server.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/calculateTransport")
    public CalculatedVehicle calculateVehicle(@RequestParam(value = "goodVehicleType",
            defaultValue = "Unknown")
                                                      String goodVehicleType,
                                              @RequestParam(value = "goodAvgConsumption",
                                                      defaultValue = "-1")
                                                      Integer goodAvgConsumption,
                                              @RequestParam(value = "badVehicleType",
                                                      defaultValue = "Unknown")
                                                      String badVehicleType,
                                              @RequestParam(value = "badAvgConsumption",
                                                      defaultValue = "-1")
                                                      Integer badAvgConsumption) {

        return new CalculatedVehicle(goodVehicleType, badVehicleType,
                goodAvgConsumption, badAvgConsumption);
    }

    /**
     the client can send data to the server with the right values as parameter,
     then this method will store the data in the database.
     */
    @RequestMapping(method = RequestMethod.POST, value = "/saveVehicleData")
    public void saveVehicleData(@RequestParam(value = "goodVehicleType",
            defaultValue = "Unknown")
                                        String goodVehicleType,
                                @RequestParam(value = "goodAvgConsumption",
                                        defaultValue = "-1")
                                        Integer goodAvgConsumption,
                                @RequestParam(value = "badVehicleType",
                                        defaultValue = "Unknown")
                                        String badVehicleType,
                                @RequestParam(value = "badAvgConsumption",
                                        defaultValue = "-1")
                                        Integer badAvgConsumption, Principal principal) {
        String username = principal.getName();
        VehicleEntry vehicleEntry = new VehicleEntry(goodVehicleType,
                badVehicleType, goodAvgConsumption, badAvgConsumption);
        Database.instance.addUsedVehicle(username, vehicleEntry);
    }

    /**
     * Given the principal, return the used vehicle list.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/usedTransportList")
    public List<VehicleEntry> getUsedVehicleList(Principal principal) {
        String username = principal.getName();
        UserEntry user =  Database.instance.findUserByName(username);
        List<VehicleEntry> usedVehicles = user.getUsedVehicles();
        return usedVehicles;
    }

}
