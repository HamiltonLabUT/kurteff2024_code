//
//  SettingsViewController.swift
//  onsetProd4kids
//
//  Created by Garret Kurteff on 1/14/21.
//  Copyright © 2021 Kurteff, Garret. All rights reserved.
//

import UIKit

class SettingsViewController: UIViewController {

//
// Initialize variables
//
    var currpid:String = "S0000" // default pid
    var currblock:String = "B1" // default block
    var currset:Bool = true // are we doing first 25 sentences (true) or last 25 (false)
    
//
// Initialize UI variables (@IBOutlets)
//

    @IBOutlet weak var pid: UITextField! // participant ID, eg. OP0001
    @IBOutlet weak var block: UITextField! // block #, B1 by default
    @IBOutlet weak var set: UISegmentedControl!
    
//
// viewDidLoad(), what happens when the Settings view is activated
//
    override func viewDidLoad() {
        super.viewDidLoad()
        pid.text = currpid
        block.text = currblock
    }
    
//
// prepare(), lets us pass values to other view controllers
//
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.destination is StartViewController {
            if set.selectedSegmentIndex == 0 {
                currset = true
                print("Will use first 25 sentences from MOCHA 4 kids.")
            }
            if set.selectedSegmentIndex == 1 {
                currset = false
                print("Will use second 25 sentences from MOCHA 4 kids.")
            }
            let vc = segue.destination as? StartViewController
            vc?.currpid = pid.text!
            vc?.currblock = block.text!
            vc?.currset = currset
        }
    }
}
